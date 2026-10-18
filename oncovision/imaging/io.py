"""Loading scans from PNG/JPG, DICOM (single file, multi-frame or zipped series) and NIfTI.

Every loader returns a :class:`Scan`: one or more co-registered channels (MRI
sequences) shaped ``(Z, H, W)``. 2D images are stored as ``Z == 1``.

Arrays are oriented in radiological convention where the format tells us how:
row 0 is anterior and column 0 is the patient's right. For plain images the
orientation is unknown and assumed radiological (``orientation_assumed``).
"""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage

# Identifying DICOM attributes removed before anything else touches the data.
PHI_TAGS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientSex",
    "PatientAge", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
    "OtherPatientNames", "PatientMotherBirthName", "MedicalRecordLocator", "EthnicGroup",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "NameOfPhysiciansReadingStudy", "RequestingPhysician", "AccessionNumber", "StudyID",
    "StudyDate", "StudyTime", "SeriesDate", "SeriesTime", "AcquisitionDate",
    "AcquisitionTime", "ContentDate", "ContentTime", "DeviceSerialNumber", "StationName",
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID",
)

SEQUENCE_ALIASES = {
    "t1ce": ("t1ce", "t1c", "t1gd", "t1_gd", "t1post", "t1-post", "contrast"),
    "flair": ("flair",),
    "t2": ("t2",),
    "t1": ("t1",),
}


class ScanLoadError(ValueError):
    """Raised when an upload cannot be parsed as a medical image."""


@dataclass
class Scan:
    channels: dict[str, np.ndarray]
    spacing: tuple[float, float, float] | None  # (z, y, x) in mm; None when unknown
    source_format: str
    metadata: dict = field(default_factory=dict)
    removed_phi: list[str] = field(default_factory=list)
    orientation_assumed: bool = False
    spacing_estimated: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return next(iter(self.channels.values())).shape

    @property
    def is_volume(self) -> bool:
        return self.shape[0] > 1

    def primary(self, preference: tuple[str, ...] = ("t1ce", "flair", "t2", "t1")) -> np.ndarray:
        for name in preference:
            if name in self.channels:
                return self.channels[name]
        return next(iter(self.channels.values()))


def guess_sequence(filename: str) -> str | None:
    """Guess the MRI sequence from a filename like ``BraTS_001_flair.nii.gz``."""
    stem = filename.lower().rsplit("/", 1)[-1]
    tokens = re.split(r"[^a-z0-9]+", stem)
    for name, aliases in SEQUENCE_ALIASES.items():
        if any(alias in tokens for alias in aliases):
            return name
    return None


# --------------------------------------------------------------------------- DICOM

def anonymize_dicom(ds) -> list[str]:
    """Strip identifying attributes and private tags in place. Returns removed tag names."""
    removed = []
    for tag in PHI_TAGS:
        if tag in ds:
            delattr(ds, tag)
            removed.append(tag)
    before = len(ds)
    ds.remove_private_tags()
    if len(ds) < before:
        removed.append(f"{before - len(ds)} private tag(s)")
    return removed


def _read_dicom(data: bytes):
    import pydicom

    try:
        return pydicom.dcmread(io.BytesIO(data), force=True)
    except Exception as exc:  # pydicom raises many error types
        raise ScanLoadError(f"Could not read DICOM: {exc}") from exc


def _dicom_pixels(ds) -> np.ndarray:
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise ScanLoadError(f"DICOM has no readable pixel data: {exc}") from exc
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3:
        arr = arr.mean(axis=-1)  # RGB secondary capture -> grayscale
    return arr if arr.ndim == 3 else arr[None]


def _dicom_metadata(ds) -> dict:
    keep = ("Modality", "BodyPartExamined", "SeriesDescription", "MagneticFieldStrength", "StudyDescription")
    return {k: str(getattr(ds, k)) for k in keep if getattr(ds, k, None) not in (None, "")}


def load_dicom_series(blobs: list[bytes]) -> Scan:
    datasets = [_read_dicom(b) for b in blobs]
    datasets = [ds for ds in datasets if "PixelData" in ds]
    if not datasets:
        raise ScanLoadError("No DICOM images with pixel data found.")

    def order(ds):
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos is not None and len(pos) == 3:
            return float(pos[2])
        return float(getattr(ds, "InstanceNumber", 0) or 0)

    datasets.sort(key=order)
    removed: set[str] = set()
    for ds in datasets:
        removed.update(anonymize_dicom(ds))

    slices = [_dicom_pixels(ds) for ds in datasets]
    base = slices[0].shape[1:]
    slices = [s for s in slices if s.shape[1:] == base]
    volume = np.concatenate(slices, axis=0)

    first = datasets[0]
    spacing = None
    pixel_spacing = getattr(first, "PixelSpacing", None)
    if pixel_spacing is not None:
        dz = float(getattr(first, "SpacingBetweenSlices", 0) or getattr(first, "SliceThickness", 0) or 1.0)
        if len(datasets) > 1:
            dz = abs(order(datasets[1]) - order(datasets[0])) or dz
        spacing = (dz, float(pixel_spacing[0]), float(pixel_spacing[1]))

    return Scan(
        channels={"image": volume},
        spacing=spacing,
        source_format="dicom",
        metadata=_dicom_metadata(first),
        removed_phi=sorted(removed),
    )


# --------------------------------------------------------------------------- NIfTI

def load_nifti(data: bytes) -> Scan:
    import nibabel as nib

    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        img = nib.Nifti1Image.from_bytes(data)
    except Exception as exc:
        raise ScanLoadError(f"Could not read NIfTI: {exc}") from exc
    img = nib.as_closest_canonical(img)  # RAS+: x->Right, y->Anterior, z->Superior
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ScanLoadError(f"Unsupported NIfTI shape {arr.shape}")
    # (X, Y, Z) -> (Z, Y, X), then flip so row 0 is anterior and column 0 is patient right.
    vol = np.transpose(arr, (2, 1, 0))[:, ::-1, ::-1].copy()
    zooms = img.header.get_zooms()[:3]
    return Scan(
        channels={"image": vol},
        spacing=(float(zooms[2]), float(zooms[1]), float(zooms[0])),
        source_format="nifti",
    )


# --------------------------------------------------------------------------- Images

def load_image(data: bytes) -> Scan:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ScanLoadError(f"Could not read image: {exc}") from exc
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.asarray(img, dtype=np.float32)
    else:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    return Scan(channels={"image": arr[None]}, spacing=None, source_format="image", orientation_assumed=True)


# --------------------------------------------------------------------------- Dispatch

def _kind(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if name.endswith(".zip") or data[:4] == b"PK\x03\x04":
        return "zip"
    if name.endswith(".dcm") or data[128:132] == b"DICM":
        return "dicom"
    return "image"


def load_file(filename: str, data: bytes) -> Scan:
    kind = _kind(filename, data)
    if kind == "nifti":
        return load_nifti(data)
    if kind == "dicom":
        return load_dicom_series([data])
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
            blobs = [zf.read(m) for m in members]
        dicoms = [b for b, m in zip(blobs, members) if _kind(m.filename, b) == "dicom" or m.filename.lower().endswith(".ima")]
        if not dicoms:
            raise ScanLoadError("ZIP archive contains no DICOM files.")
        return load_dicom_series(dicoms)
    return load_image(data)


def _match_shape(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    factors = [t / s for t, s in zip(shape, arr.shape)]
    return ndimage.zoom(arr, factors, order=1)[: shape[0], : shape[1], : shape[2]]


def load_scan(files: list[tuple[str, bytes, str | None]]) -> Scan:
    """Load one study from uploaded files.

    ``files`` is a list of ``(filename, bytes, sequence_label)``. Several files
    with different sequence labels (or filenames like ``*_flair.nii.gz``) are
    combined into one multi-sequence scan. Several unlabeled DICOM files are
    treated as slices of a single series.
    """
    if not files:
        raise ScanLoadError("No files uploaded.")

    unlabeled_dicoms = [f for f in files if not f[2] and _kind(f[0], f[1]) == "dicom" and not guess_sequence(f[0])]
    if len(unlabeled_dicoms) == len(files) and len(files) > 1:
        return load_dicom_series([f[1] for f in files])

    combined: Scan | None = None
    for index, (filename, data, label) in enumerate(files):
        scan = load_file(filename, data)
        name = (label or guess_sequence(filename) or ("image" if index == 0 else f"image_{index + 1}")).lower()
        arr = scan.channels["image"]
        if combined is None:
            combined = scan
            combined.channels = {name: arr}
            continue
        if name in combined.channels:
            raise ScanLoadError(f"Two files were labeled as the same sequence '{name}'.")
        combined.channels[name] = _match_shape(arr, combined.shape)
        combined.removed_phi = sorted(set(combined.removed_phi) | set(scan.removed_phi))
    assert combined is not None
    return combined
