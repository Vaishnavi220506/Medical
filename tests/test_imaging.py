import io
import zipfile

import nibabel as nib
import numpy as np
import pydicom
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
    m = measure(disk, ref, get_profile("breast_us"), (1.0, 0.5, 0.5), axial_volume=False)
    assert abs(m["longest_diameter"] - 20.5) < 1.0  # 41 px * 0.5 mm
    assert abs(m["perpendicular_diameter"] - 20.5) < 1.5
    assert abs(m["total"]["area_mm2"] - np.pi * 10**2) < 15
    assert m["location"]["side"] == "patient right"  # image left
