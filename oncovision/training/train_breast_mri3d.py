"""Train a 3D breast tumor segmenter (3D U-Net) on breast DCE-MRI from MAMA-MIA.

MAMA-MIA (Garrucho et al., Scientific Data 2025): 1,506 breast cancer patients from four
collections (Duke, ISPY1, ISPY2, NACT), each with the first post-contrast MRI and an expert
3D outline of the primary tumor. The Hugging Face redistribution used here
(YongchengYAO/MAMA-MIA-Lite) is CC BY-NC 4.0: research and education only.

Each scan is scaled to [0, 1] and resampled to 1.5 mm like the app does, then cropped to
the body. Training (oncovision.training.lesion3d) uses 3D patches, half of them centred on
the tumor. Needs a GPU (a free Kaggle T4 works); see notebooks/train_breast_mri3d.ipynb.

    python -m oncovision.training.train_breast_mri3d --prepared-dir /tmp/breast3d --epochs 60

The archive stores files uncompressed, so each case is read straight out of the remote
zip; nothing large is downloaded.
"""

from __future__ import annotations

import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import settings
from ..imaging.io import load_nifti
from ..imaging.preprocess import normalize
from .lesion3d import add_training_args, train, write_splits

HF_REPO = "YongchengYAO/MAMA-MIA-Lite"
PARTS = ("data-part001.zip", "data-part002.zip")
SPACING = 1.5
MIN_TUMOR_MM3 = 30  # about a 4 mm sphere
MARGIN = 4


def to_model_space(vol: np.ndarray, spacing) -> np.ndarray:
    """MRI volume -> [0, 1] (like the app's channel_stack) on a 1.5 mm grid."""
    x = normalize(vol)
    factors = [s / SPACING for s in spacing]
    return ndimage.zoom(x, factors, order=1) if any(abs(f - 1) > 0.05 for f in factors) else x


def body_box(x: np.ndarray) -> tuple[slice, ...]:
    """Same crop as the app's 3D path: everything brighter than background, plus a margin."""
    fg = x > 0.02
    if not fg.any():
        return tuple(slice(0, s) for s in x.shape)
    idx = [np.where(fg.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    return tuple(slice(max(i[0] - MARGIN, 0), i[-1] + MARGIN + 1) for i in idx)


def prepare_case(image_bytes: bytes, mask_bytes: bytes, target: Path) -> str:
    scan = load_nifti(image_bytes)
    mask = load_nifti(mask_bytes).channels["image"] > 0.5
    x = to_model_space(scan.channels["image"], scan.spacing)
    m = ndimage.zoom(mask.astype(np.float32), [a / b for a, b in zip(x.shape, mask.shape)], order=1) > 0.5
    box = body_box(x)
    np.savez_compressed(target, image=(x[box] * 255).astype(np.uint8), label=m[box].astype(np.uint8))
    return f"{target.stem}: {x[box].shape}, tumor voxels {int(m.sum())}"


def _job(args) -> str:
    part_img, part_mask, case, target = args
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        blobs = []
        for part, member in ((part_img, f"Images/{case}.nii.gz"), (part_mask, f"Masks/{case}.nii.gz")):
            with fs.open(f"datasets/{HF_REPO}/{part}", "rb", block_size=1 << 22) as f, zipfile.ZipFile(f) as z:
                blobs.append(z.read(member))
        return prepare_case(*blobs, Path(target))
    except Exception as exc:  # one broken case should not stop the whole run
        return f"skip {case}: {exc}"


def prepare(out: Path, workers: int, max_cases: int | None) -> list[Path]:
    from huggingface_hub import HfFileSystem

    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    where: dict[str, dict[str, str]] = {}
    for part in PARTS:
        with fs.open(f"datasets/{HF_REPO}/{part}", "rb", block_size=1 << 20) as f, zipfile.ZipFile(f) as z:
            for name in z.namelist():
                if name.endswith(".nii.gz"):
                    kind, file = name.split("/", 1)
                    where.setdefault(file.removesuffix(".nii.gz"), {})[kind] = part
    cases = sorted(c for c, d in where.items() if {"Images", "Masks"} <= set(d))[:max_cases]
    jobs = [(where[c]["Images"], where[c]["Masks"], c, str(out / f"{c}.npz")) for c in cases if not (out / f"{c}.npz").exists()]
    print(f"preparing {len(jobs)} of {len(cases)} cases", flush=True)
    with ProcessPoolExecutor(max(workers, 1)) as pool:
        for n, msg in enumerate(pool.map(_job, jobs), 1):
            print(f"[{n}/{len(jobs)}] {msg}", flush=True)
    return sorted(out.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(parser, val_cases=60, test_cases=100)
    args = parser.parse_args()
    files = prepare(args.prepared_dir, args.workers, args.max_cases)
    write_splits(args.prepared_dir, args.val_cases, args.test_cases)
    if args.prepare_only:
        return
    meta = {
        "profile": "breast_mri",
        "spacing_mm": SPACING,
        "crop": "foreground",
        "min_component_mm3": MIN_TUMOR_MM3,
        "data": "MAMA-MIA (YongchengYAO/MAMA-MIA-Lite, CC BY-NC 4.0)",
    }
    out = args.out or settings.weights_dir / "breast_mri" / "segmenter3d.pt"
    train(args, files, out, meta, None, max(1, round(MIN_TUMOR_MM3 / SPACING**3)))


if __name__ == "__main__":
    main()
