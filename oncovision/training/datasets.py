"""Dataset readers for the public datasets OncoVision is trained on.

All readers produce images normalized exactly like inference does
(``oncovision.imaging``), so a model sees the same kind of input in training
and in the app. See docs/DATASETS.md for download links and folder layouts.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..imaging.io import load_nifti
from ..imaging.preprocess import normalize, resize2d

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Folder names used by public datasets -> our class names.
CLASS_ALIASES = {
    "notumor": "no_tumor", "notumour": "no_tumor", "no": "no_tumor", "notumorcases": "no_tumor",
    "glioma": "glioma", "gliomatumor": "glioma",
    "meningioma": "meningioma", "meningiomatumor": "meningioma",
    "pituitary": "pituitary", "pituitarytumor": "pituitary",
    "normal": "normal", "normalcases": "normal",
    "benign": "benign", "bengincases": "benign", "benigncases": "benign",
    "malignant": "malignant", "malignantcases": "malignant",
}


def canonical_class(folder: str) -> str:
    key = re.sub(r"[^a-z]", "", folder.lower())
    return CLASS_ALIASES.get(key, folder.lower())


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float32) if img.mode.startswith("I") else np.asarray(img.convert("L"), dtype=np.float32)
    return normalize(arr)


def _is_mask(path: Path) -> bool:
    return "_mask" in path.stem.lower()


# ---------------------------------------------------------------------- classification

def find_class_images(root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    """All images under ``root/<class folder>/...`` whose folder maps to one of ``classes``."""
    items = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        name = canonical_class(folder.name)
        if name not in classes:
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXT and not _is_mask(path):
                items.append((path, classes.index(name)))
    return items


def split_items(items: list, val_fraction: float, seed: int = 42) -> tuple[list, list]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = int(len(items) * val_fraction)
    return items[n_val:], items[:n_val]


class ClassificationDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], image_size: int = 224):
        self.items = items
        self.size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = resize2d(_load_gray(path), (self.size, self.size))
        return torch.from_numpy(img[None].copy()).float(), label


# ---------------------------------------------------------------------- 2D segmentation

def find_busi_pairs(root: Path) -> list[tuple[Path, list[Path]]]:
    """BUSI: ``benign (1).png`` with ``benign (1)_mask.png`` (and sometimes ``_mask_1.png``)."""
    pairs = []
    for img in sorted(Path(root).rglob("*.png")):
        if _is_mask(img):
            continue
        masks = sorted(img.parent.glob(f"{img.stem}_mask*.png"))
        if masks:
            pairs.append((img, masks))
    return pairs


def find_mask_folder_pairs(images: Path, masks: Path) -> list[tuple[Path, list[Path]]]:
    """Generic layout: ``images/<name>.<ext>`` with binary ``masks/<name>.png``."""
    lookup = {p.stem: p for p in Path(masks).rglob("*") if p.suffix.lower() in IMAGE_EXT}
    return [(p, [lookup[p.stem]]) for p in sorted(Path(images).rglob("*")) if p.suffix.lower() in IMAGE_EXT and p.stem in lookup]


class MaskSegmentationDataset(Dataset):
    """Single-channel images with binary masks (0 / 255): 1 = tumor, 0 = background."""

    def __init__(self, pairs, image_size: int = 192):
        self.pairs = pairs
        self.size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_paths = self.pairs[i]
        img = resize2d(_load_gray(img_path), (self.size, self.size))
        mask = np.zeros((self.size, self.size), dtype=np.int64)
        for mp in mask_paths:
            m = np.asarray(Image.open(mp).convert("L"))
            m = resize2d(m.astype(np.float32), (self.size, self.size), order=0)
            mask = np.maximum(mask, (m > 127).astype(np.int64))
        return torch.from_numpy(img[None].copy()).float(), torch.from_numpy(mask)


# ---------------------------------------------------------------------- BraTS (3D -> 2D slices)

BRATS_SEQUENCES = ("flair", "t1", "t1ce", "t2")


def _brats_file(case: Path, key: str) -> Path | None:
    for pattern in (f"*_{key}.nii.gz", f"*_{key}.nii", f"*-{key}.nii.gz"):
        hits = sorted(case.glob(pattern))
        if hits:
            return hits[0]
    return None


def prepare_brats(root: Path, out: Path, size: int = 192, background_fraction: float = 0.25, seed: int = 42) -> int:
    """Convert BraTS cases into 2D slice files ``out/<case>_<z>.npz``.

    Keeps every slice containing tumor plus a random share of tumor-free brain
    slices. Labels are remapped to OncoVision's: 1 necrotic core, 2 edema,
    3 enhancing tumor (BraTS 2021 uses 4 for enhancing; BraTS 2023 uses 3).
    BraTS 2023 file names (t1n, t1c, t2w, t2f) are also accepted.
    """
    rng = random.Random(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    aliases = {"flair": ("flair", "t2f"), "t1": ("t1", "t1n"), "t1ce": ("t1ce", "t1c"), "t2": ("t2", "t2w")}
    written = 0
    cases = sorted(p for p in Path(root).rglob("*") if p.is_dir() and (_brats_file(p, "seg") is not None))
    for n, case in enumerate(cases, 1):
        paths = {}
        for seq, names in aliases.items():
            paths[seq] = next((f for name in names if (f := _brats_file(case, name)) is not None), None)
        if any(v is None for v in paths.values()):
            print(f"skip {case.name}: missing sequence")
            continue
        vols = np.stack([normalize(load_nifti(paths[s].read_bytes()).channels["image"]) for s in BRATS_SEQUENCES])
        seg = load_nifti(_brats_file(case, "seg").read_bytes()).channels["image"].round().astype(np.uint8)
        seg[seg == 4] = 3
        for z in range(seg.shape[0]):
            has_tumor = seg[z].any()
            has_brain = vols[:, z].max() > 0.05
            if not has_brain or (not has_tumor and rng.random() > background_fraction):
                continue
            img = (resize2d(vols[:, z], (size, size)) * 255).astype(np.uint8)
            lab = resize2d(seg[z].astype(np.float32), (size, size), order=0).astype(np.uint8)
            np.savez_compressed(out / f"{case.name}_{z:03d}.npz", image=img, label=lab)
            written += 1
        if n % 25 == 0:
            print(f"prepared {n}/{len(cases)} cases, {written} slices")
    return written


class SliceFileDataset(Dataset):
    """Prepared BraTS slices (C x H x W uint8 images, H x W labels)."""

    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as data:
            img = data["image"].astype(np.float32) / 255.0
            lab = data["label"].astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(lab)


def split_by_key(items: list, key, val_fraction: float, seed: int = 42) -> tuple[list, list]:
    """Split so all items sharing ``key(item)`` (e.g. a patient) land on the same side."""
    groups = sorted({key(it) for it in items})
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups) - 1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups[: max(n_val, 0)])
    return [it for it in items if key(it) not in val_groups], [it for it in items if key(it) in val_groups]


def split_by_case(files: list[Path], val_fraction: float, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Split slice files so no patient appears in both train and validation."""
    cases = sorted({f.stem.rsplit("_", 1)[0] for f in files})
    random.Random(seed).shuffle(cases)
    n_val = min(len(cases) - 1, max(1, int(len(cases) * val_fraction)))  # always keep a training case
    val_cases = set(cases[:n_val])
    train = [f for f in files if f.stem.rsplit("_", 1)[0] not in val_cases]
    val = [f for f in files if f.stem.rsplit("_", 1)[0] in val_cases]
    return train, val
