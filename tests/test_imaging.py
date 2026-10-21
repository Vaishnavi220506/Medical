import io
import zipfile

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from conftest import png_bytes
from oncovision.imaging.io import guess_sequence, load_scan
from oncovision.imaging.measure import measure
from oncovision.models.registry import get_profile


def _dicom(z: int, value: int = 100) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("x.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "123456"
    ds.PatientBirthDate = "19800101"
    ds.InstitutionName = "General Hospital"
    ds.Modality = "MR"
    ds.SeriesDescription = "AX T1 POST"
    ds.Rows = ds.Columns = 32
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3
    ds.ImagePositionPatient = [0, 0, z * 3.0]
    ds.InstanceNumber = z + 1
    ds.PixelData = (np.full((32, 32), value + z, dtype=np.uint16)).tobytes()
    ds.add_new(0x00091010, "LO", "private vendor data")
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_png_is_single_slice():
    scan = load_scan([("x.png", png_bytes(np.arange(64 * 64).reshape(64, 64) % 255), None)])
    assert scan.shape == (1, 64, 64)
    assert scan.orientation_assumed and scan.spacing is None


def test_nifti_is_radiological():
    # RAS volume with a bright marker on the patient's right (+x) and anterior (+y) side.
    arr = np.zeros((20, 20, 5), dtype=np.float32)
    arr[17, 17, 2] = 1
    data = nib.Nifti1Image(arr, np.diag([2.0, 2.0, 4.0, 1.0])).to_bytes()
    scan = load_scan([("brain_flair.nii", data, None)])
    z, row, col = np.argwhere(scan.channels["flair"] == 1)[0]
    assert (z, row, col) == (2, 2, 2)  # anterior -> top rows, patient right -> left columns
    assert scan.spacing == (4.0, 2.0, 2.0)


def test_dicom_series_is_stacked_and_anonymized():
    blobs = [_dicom(z) for z in (2, 0, 1)]
    scan = load_scan([(f"img{i}.dcm", b, None) for i, b in enumerate(blobs)])
    assert scan.shape == (3, 32, 32)
    assert [scan.channels["image"][i, 0, 0] for i in range(3)] == [100, 101, 102]  # sorted by position
    assert {"PatientName", "PatientID", "PatientBirthDate", "InstitutionName"} <= set(scan.removed_phi)
    assert scan.spacing == (3.0, 0.5, 0.5)
    assert scan.metadata["Modality"] == "MR"
    assert "Doe" not in str(scan.metadata)


def test_dicom_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for z in range(4):
            zf.writestr(f"series/IM{z}", _dicom(z))
    scan = load_scan([("series.zip", buf.getvalue(), None)])
    assert scan.shape == (4, 32, 32)


def test_multi_sequence_naming():
    assert guess_sequence("BraTS2021_00001_t1ce.nii.gz") == "t1ce"
    assert guess_sequence("case_FLAIR.nii") == "flair"
    assert guess_sequence("BraTS-GLI-00000-000-t1c.nii.gz") == "t1ce"
    assert guess_sequence("scan.png") is None


def test_measurements_of_known_disk():
    yy, xx = np.mgrid[:200, :200]
    disk = ((yy - 100) ** 2 + (xx - 60) ** 2 <= 20**2).astype(np.uint8)[None]
    ref = np.zeros((1, 200, 200), dtype=np.float32)
    ref[0, 20:180, 20:180] = 1
    m = measure(disk, ref, get_profile("lung_ct"), (1.0, 0.5, 0.5), axial_volume=False)
    assert abs(m["longest_diameter"] - 20.5) < 1.0  # 41 px * 0.5 mm
    assert abs(m["perpendicular_diameter"] - 20.5) < 1.5
    assert abs(m["total"]["area_mm2"] - np.pi * 10**2) < 15
    assert m["location"]["side"] == "patient right"  # image left


def test_ct_window_applies_only_to_hounsfield_data():
    from oncovision.analysis.pipeline import apply_window
    from oncovision.imaging.io import Scan

    hu = Scan(channels={"image": np.array([[[-2000.0, 0.0, 3000.0]]])}, spacing=None, source_format="nifti")
    apply_window(hu, get_profile("lung_ct"))
    assert hu.channels["image"].tolist() == [[[-1000.0, 0.0, 400.0]]]
    png = Scan(channels={"image": np.array([[[0.0, 255.0]]])}, spacing=None, source_format="image")
    apply_window(png, get_profile("lung_ct"))
    assert png.channels["image"].tolist() == [[[0.0, 255.0]]]


def test_pseudo_3d_inflation():
    import numpy as np

    from oncovision.imaging.mesh import _inflate, build_pseudo_meshes
    from oncovision.models.registry import PROFILES

    yy, xx = np.mgrid[:128, :128]
    big = (yy - 60) ** 2 + (xx - 40) ** 2 <= 30**2
    small = (yy - 60) ** 2 + (xx - 105) ** 2 <= 5**2
    vol = _inflate(big | small, 80)
    # A disk becomes a ball: depth ~ diameter. The small disk stays small.
    assert abs(vol[:, 60, 40].sum() - 60) <= 2
    assert vol[:, 60, 105].sum() <= 12

    img = np.zeros((1, 128, 128), np.float32)
    img[0][(yy - 64) ** 2 + (xx - 64) ** 2 <= 55**2] = 0.5
    lab = np.zeros((1, 128, 128), np.uint8)
    lab[0][(yy - 64) ** 2 + (xx - 64) ** 2 <= 15**2] = 4
    out = build_pseudo_meshes(img, lab, PROFILES["brain_mri"], (1.0, 0.5, 0.5))
    assert out["pseudo"] and [m["name"] for m in out["meshes"]] == ["anatomy", "possible_extent", "tumor"]


def _chest(liver_at_end: bool) -> np.ndarray:
    """Synthetic whole-chest breast MRI (40 x 128 x 140, 4 x 2.5 x 2.5 mm): torso, two breasts,
    dark lungs, and optionally a liver (medium grey) on the patient's right at the last slices."""
    z, r, c = np.ogrid[:40, :128, :140]
    vol = np.zeros((40, 128, 140), np.float32)
    vol[:] = np.where(((r - 90) / 30.0) ** 2 + ((c - 70) / 65.0) ** 2 <= 1, 0.3, 0)  # torso
    for col in (38, 102):  # breasts in front of the chest wall
        vol[((z - 20) * 4 / 60.0) ** 2 + ((r - 66) / 14.0) ** 2 + ((c - col) / 20.0) ** 2 <= 1] = 0.3
    vol[:, 88:110, 20:62] = vol[:, 88:110, 78:120] = 0.02  # lungs
    if liver_at_end:
        vol[30:, 88:115, 15:65] = 0.6
    return vol


def test_breast_figure_and_head_end():
    from oncovision.imaging.mesh import breast_head_first, build_meshes

    profile = get_profile("breast_mri")
    labels = np.zeros((40, 128, 140), np.uint8)
    labels[16:24, 58:72, 32:44] = 1
    for liver, expected in ((False, False), (True, True)):
        vol = _chest(liver)
        assert breast_head_first(vol, (4.0, 2.5, 2.5)) is expected
        data = build_meshes(vol, labels, profile, (4.0, 2.5, 2.5))
        names = [m["name"] for m in data["meshes"]]
        assert data["anatomy_label"] == "Breasts" and "body" in names and data["flip_vertical"] is expected
        assert next(m for m in data["meshes"] if m["name"] == "tumor")["spots"]
    # One breast only (narrow scan): no illustrated body, no flip.
    narrow = build_meshes(_chest(True)[:, :, :100], labels[:, :, :100], profile, (4.0, 2.5, 2.5))
    assert "body" not in [m["name"] for m in narrow["meshes"]] and "flip_vertical" not in narrow
