"""Turn LIDC-IDRI CT volumes with nodule masks into 2D training data for the lung models.

Expects NIfTI volumes and masks, e.g. from https://huggingface.co/datasets/totalol/lungct::

    <raw>/malignant_cases_CT/LIDC-IDRI-0001_0000.nii.gz
    <raw>/malignant_cases_CT/MASK/LIDC-IDRI-0001.nii.gz
    <raw>/benign_cases_CT1/...

Writes (lung window -1000..400 HU, 256x256, radiological orientation)::

    <out>/seg/images/<case>__<z>.png  + <out>/seg/masks/<case>__<z>.png
    <out>/cls/{normal,benign,malignant}/<case>__<z>.png

Nodule slices take the benign/malignant label of their case folder. "normal"
slices are lung slices at least 10 slices away from any nodule. Some normal
slices are also added to the segmentation set (empty masks) so the segmenter
learns not to fire on healthy lung.

Usage:  python scripts/prepare_lidc.py --raw data/lidc --out data/lidc_2d
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oncovision.imaging.io import load_nifti  # noqa: E402
from oncovision.imaging.preprocess import resize2d  # noqa: E402

WINDOW = (-1000.0, 400.0)
SIZE = 256


def _lung_fraction(sl: np.ndarray) -> float:
    """Share of the body cross-section that is air (lung), ignoring air outside the body."""
    body = ndimage.binary_fill_holes(ndimage.binary_closing(sl > 60, iterations=3))
    body = ndimage.binary_erosion(body, iterations=4)
    return float(((sl < 60) & body).sum() / max(body.sum(), 1))


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normal-per-case", type=int, default=6)
    parser.add_argument(
        "--no-reorient",
        action="store_true",
        help="Skip the axis fix. The Hugging Face mirror's NIfTI headers swap x/y, so by default each slice "
        "is rotated and mirrored into standard radiological orientation (spine at the bottom, liver on the image left).",
    )
    args = parser.parse_args()
    rng = random.Random(0)

    volumes = sorted(p for p in args.raw.rglob("*_0000.nii.gz"))
    counts: dict[str, int] = {}
    for n, vol_path in enumerate(volumes, 1):
        case = vol_path.name.replace("_0000.nii.gz", "")
        mask_path = next((p for p in vol_path.parent.iterdir() if p.is_dir() and p.name.lower() == "mask"), None)
        mask_file = mask_path / f"{case}.nii.gz" if mask_path else None
        if not mask_file or not mask_file.exists():
            print(f"skip {case}: no mask")
            continue
        label = "malignant" if "malignant" in vol_path.parent.name.lower() else "benign"
        vol = load_nifti(vol_path.read_bytes()).channels["image"]
        mask = load_nifti(mask_file.read_bytes()).channels["image"] > 0.5
        if not args.no_reorient:
            vol = np.ascontiguousarray(np.rot90(vol, 1, axes=(1, 2))[:, :, ::-1])
            mask = np.ascontiguousarray(np.rot90(mask, 1, axes=(1, 2))[:, :, ::-1])
        img = ((np.clip(vol, *WINDOW) - WINDOW[0]) / (WINDOW[1] - WINDOW[0]) * 255).astype(np.uint8)

        nodule_z = np.where(mask.any(axis=(1, 2)))[0]
        for z in nodule_z:
            im = resize2d(img[z].astype(np.float32), (SIZE, SIZE)).clip(0, 255).astype(np.uint8)
            mk = (resize2d(mask[z].astype(np.float32), (SIZE, SIZE), order=0) > 0.5).astype(np.uint8) * 255
            if not mk.any():
                continue
            name = f"{case}__{z:03d}.png"
            _save(im, args.out / "seg" / "images" / name)
            _save(mk, args.out / "seg" / "masks" / name)
            _save(im, args.out / "cls" / label / name)
            counts[label] = counts.get(label, 0) + 1

        # Healthy lung slices: inside the lung range, far from any nodule.
        lungs = [z for z in range(img.shape[0]) if _lung_fraction(img[z]) > 0.15]
        far = [z for z in lungs if nodule_z.size == 0 or np.abs(nodule_z - z).min() >= 10]
        rng.shuffle(far)
        for i, z in enumerate(far[: args.normal_per_case]):
            im = resize2d(img[z].astype(np.float32), (SIZE, SIZE)).clip(0, 255).astype(np.uint8)
            name = f"{case}__{z:03d}.png"
            _save(im, args.out / "cls" / "normal" / name)
            counts["normal"] = counts.get("normal", 0) + 1
            if i < args.normal_per_case // 2:
                _save(im, args.out / "seg" / "images" / name)
                _save(np.zeros((SIZE, SIZE), np.uint8), args.out / "seg" / "masks" / name)
        print(f"[{n}/{len(volumes)}] {case} ({label}): {nodule_z.size} nodule slices")
    print(f"Done: {counts} -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
