"""Cut varied 2D training slices from BraTS volumes for the single-image brain segmenter.

The Figshare data is all contrast-enhanced T1 in mostly one orientation, so a
segmenter trained only on it misses tumors on FLAIR / T2 images and other views.
This script slices BraTS 3D scans in all three orientations (axial, coronal,
sagittal) and three sequences (T1ce, FLAIR, T2):

* T1ce slices are labeled with the tumor core (necrosis + enhancing tumor),
  which is what is visible on contrast T1 (and what Figshare masks show).
* FLAIR and T2 slices are labeled with the whole tumor including edema,
  which is what is bright on those sequences.
* A tumor-free slice per case is added with an empty mask.

Output uses the ``<patient>__<name>.png`` convention so training splits by patient::

    <out>/images/<patient>__<case>_<view>_<seq>_<index>.png
    <out>/masks/...

Usage:  python scripts/prepare_brats_slices2d.py --data data/BraTS2023 --out data/brats2d
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oncovision.imaging.io import load_nifti  # noqa: E402
from oncovision.imaging.preprocess import normalize  # noqa: E402
from oncovision.training.datasets import _brats_file, patient_id  # noqa: E402

SEQUENCES = {"t1ce": ("t1ce", "t1c"), "flair": ("flair", "t2f"), "t2": ("t2", "t2w")}


def _square(arr: np.ndarray) -> np.ndarray:
    """Zero-pad to a centred square so resizing later doesn't distort anatomy."""
    h, w = arr.shape
    n = max(h, w)
    out = np.zeros((n, n), dtype=arr.dtype)
    out[(n - h) // 2 : (n - h) // 2 + h, (n - w) // 2 : (n - w) // 2 + w] = arr
    return out


def _views(vol: np.ndarray):
    """(name, slicer) for axial, coronal and sagittal slices in display orientation."""
    yield "axial", vol.shape[0], lambda a, i: a[i]
    yield "coronal", vol.shape[1], lambda a, i: np.flipud(a[:, i, :])  # superior at the top
    yield "sagittal", vol.shape[2], lambda a, i: np.flipud(a[:, :, i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-view", type=int, default=3, help="Tumor slices per case and orientation")
    args = parser.parse_args()
    rng = random.Random(0)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)

    cases = sorted(p for p in args.data.iterdir() if p.is_dir() and _brats_file(p, "seg") is not None)
    written = 0
    for n, case in enumerate(cases, 1):
        vols = {}
        for seq, names in SEQUENCES.items():
            f = next((f for name in names if (f := _brats_file(case, name)) is not None), None)
            if f is not None:
                vols[seq] = (normalize(load_nifti(f.read_bytes()).channels["image"]) * 255).astype(np.uint8)
        if not vols:
            continue
        seg = load_nifti(_brats_file(case, "seg").read_bytes()).channels["image"].round().astype(np.uint8)
        masks = {"core": np.isin(seg, (1, 3, 4)), "whole": seg > 0}
        pid = patient_id(case.name)

        def save(img, mask, name):
            nonlocal written
            stem = f"{pid}__{case.name}_{name}.png"
            Image.fromarray(_square(img)).save(args.out / "images" / stem)
            Image.fromarray(_square(mask.astype(np.uint8) * 255)).save(args.out / "masks" / stem)
            written += 1

        brain = vols[next(iter(vols))] > 20
        for view, count, take in _views(seg):
            # Tumor slices that show a substantial brain cross-section (not the very edge).
            tumor_idx = [i for i in range(count) if take(masks["whole"], i).sum() > 100 and take(brain, i).mean() > 0.12]
            if not tumor_idx:
                continue
            for k, i in enumerate(rng.sample(tumor_idx, min(args.per_view, len(tumor_idx)))):
                seq = rng.choice(sorted(vols))
                mask = take(masks["core"] if seq == "t1ce" else masks["whole"], i)
                save(take(vols[seq], i), mask, f"{view}_{seq}_{k}")
        # One tumor-free axial slice with brain in it.
        free = [i for i in range(seg.shape[0]) if not masks["whole"][i].any() and (vols[next(iter(vols))][i] > 20).mean() > 0.15]
        if free:
            i = rng.choice(free)
            seq = rng.choice(sorted(vols))
            save(vols[seq][i], np.zeros_like(masks["whole"][i]), f"axial_{seq}_neg")
        if n % 50 == 0:
            print(f"{n}/{len(cases)} cases, {written} slices", flush=True)
    print(f"Wrote {written} slices from {len(cases)} cases to {args.out.resolve()}")


if __name__ == "__main__":
    main()
