"""The cancer types OncoVision knows about.

Each :class:`CancerProfile` describes one scan type: which classes its
classifier predicts, which tumor sub-regions its segmenter outlines, and how
inputs should be prepared. Adding a new cancer means adding a profile here and
training its two models with ``oncovision.training``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegLabel:
    index: int
    name: str
    color: tuple[int, int, int]
    description: str


@dataclass(frozen=True)
class CancerProfile:
    key: str
    display_name: str
    organ: str
    modality: str
    classes: tuple[str, ...]
    class_descriptions: dict[str, str]
    normal_class: str
    seg_labels: tuple[SegLabel, ...]
    # MRI sequences the segmenter expects, in channel order. Missing sequences
    # are filled with an available one (models are trained with sequence dropout).
    seg_channels: tuple[str, ...] = ("image",)
    classifier_channel_preference: tuple[str, ...] = ("image",)
    image_size: int = 224
    seg_size: int = 192
    # Whether lesions are usually brighter ("bright") or darker ("dark") than
    # surrounding tissue; used by the heuristic demo segmenter.
    lesion_polarity: str = "bright"
    reporting_system: str = ""
    datasets: tuple[str, ...] = field(default_factory=tuple)

    @property
    def num_seg_classes(self) -> int:
        return len(self.seg_labels) + 1  # + background

    def label(self, index: int) -> SegLabel:
        return next(lbl for lbl in self.seg_labels if lbl.index == index)


PROFILES: dict[str, CancerProfile] = {
    "brain_mri": CancerProfile(
        key="brain_mri",
        display_name="Brain tumor (MRI)",
        organ="brain",
        modality="MRI",
        classes=("no_tumor", "glioma", "meningioma", "pituitary"),
        class_descriptions={
            "no_tumor": "No tumor detected",
            "glioma": "Glioma: a tumor arising from glial (supporting) cells of the brain",
            "meningioma": "Meningioma: a usually slow-growing tumor of the meninges covering the brain",
            "pituitary": "Pituitary tumor: a growth in the pituitary gland at the skull base",
        },
        normal_class="no_tumor",
        seg_labels=(
            SegLabel(1, "necrotic_core", (235, 64, 52), "Necrotic / non-enhancing tumor core"),
            SegLabel(2, "edema", (250, 204, 21), "Peritumoral edema (swelling)"),
            SegLabel(3, "enhancing_tumor", (59, 130, 246), "Enhancing (active) tumor"),
        ),
        seg_channels=("flair", "t1", "t1ce", "t2"),
        classifier_channel_preference=("t1ce", "t1", "flair", "t2", "image"),
        lesion_polarity="bright",
        reporting_system="RANO (Response Assessment in Neuro-Oncology)",
        datasets=("Kaggle Brain Tumor MRI (classification)", "BraTS 2021 (segmentation)"),
    ),
    "breast_us": CancerProfile(
        key="breast_us",
        display_name="Breast lesion (ultrasound)",
        organ="breast",
        modality="Ultrasound",
        classes=("normal", "benign", "malignant"),
        class_descriptions={
            "normal": "No lesion detected",
            "benign": "Benign-appearing lesion (e.g. fibroadenoma, cyst)",
            "malignant": "Lesion with malignant features",
        },
        normal_class="normal",
        seg_labels=(SegLabel(1, "lesion", (236, 72, 153), "Breast lesion"),),
        lesion_polarity="dark",
        reporting_system="BI-RADS (Breast Imaging Reporting and Data System)",
        datasets=("BUSI - Breast Ultrasound Images Dataset",),
    ),
    "lung_ct": CancerProfile(
        key="lung_ct",
        display_name="Lung nodule (CT)",
        organ="lung",
        modality="CT",
        classes=("normal", "benign", "malignant"),
        class_descriptions={
            "normal": "No nodule detected",
            "benign": "Benign-appearing nodule",
            "malignant": "Nodule with malignant features",
        },
        normal_class="normal",
        seg_labels=(SegLabel(1, "nodule", (16, 185, 129), "Lung nodule"),),
        lesion_polarity="bright",
        reporting_system="Lung-RADS",
        datasets=("IQ-OTH/NCCD lung cancer CT (classification)", "LIDC-IDRI (nodule masks)"),
    ),
}

# Scan types the router can recognise, including "other" for unsupported scans.
ROUTER_CLASSES: tuple[str, ...] = (*PROFILES.keys(), "other")


def get_profile(key: str) -> CancerProfile:
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown scan type '{key}'. Known: {', '.join(PROFILES)}") from exc
