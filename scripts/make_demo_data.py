"""Generate synthetic demo scans so OncoVision can be tried without real data.

Creates in ``demo_data/``:
  brain_t0_{flair,t1,t1ce,t2}.nii.gz   phantom brain MRI with a tumor (baseline)
  brain_t1_{flair,t1,t1ce,t2}.nii.gz   same phantom 3 months later, tumor grown
  brain_slice.png                      a single 2D slice (like Kaggle images)
  breast_us.png                        phantom breast ultrasound with a dark lesion

These are simple geometric phantoms for testing the software, not real anatomy.

Usage:  python scripts/make_demo_data.py [--out demo_data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

SHAPE = (128, 128, 80)  # x, y, z (RAS voxel grid, 1.5 mm)
SPACING = 1.5

# Signal intensity of each tissue per sequence (roughly realistic contrasts).
TISSUE = {
    #          flair  t1    t1ce   t2
    "scalp":   (0.35, 0.70, 0.70, 0.40),
    "skull":   (0.05, 0.10, 0.10, 0.05),
    "brain":   (0.40, 0.55, 0.55, 0.40),
    "csf":     (0.08, 0.15, 0.15, 0.90),
    "edema":   (0.85, 0.40, 0.42, 0.80),
    "enhance": (0.65, 0.45, 0.95, 0.60),
    "necrosis": (0.30, 0.20, 0.20, 0.85),
}
SEQUENCES = ("flair", "t1", "t1ce", "t2")


def _ellipsoid(grid, center, radii):
    x, y, z = grid
    return ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 + ((z - center[2]) / radii[2]) ** 2 <= 1


def phantom(tumor_radius: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    grid = np.mgrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]].astype(np.float32)
    c = (64, 64, 40)
    head = _ellipsoid(grid, c, (52, 60, 38))
    skull_outer = _ellipsoid(grid, c, (48, 56, 35))
    brain = _ellipsoid(grid, c, (44, 52, 32))
    ventricles = _ellipsoid(grid, (58, 66, 44), (4, 14, 6)) | _ellipsoid(grid, (70, 66, 44), (4, 14, 6))

    # Tumor in the right frontal region (RAS grid: +x = patient right, +y = anterior).
    t_center = (88, 84, 46)
    enhancing = _ellipsoid(grid, t_center, (tumor_radius,) * 3) & brain
    necrosis = _ellipsoid(grid, t_center, (tumor_radius * 0.55,) * 3) & brain
    edema = _ellipsoid(grid, t_center, (tumor_radius * 1.9, tumor_radius * 1.7, tumor_radius * 1.6)) & brain

    vols = {}
    for i, seq in enumerate(SEQUENCES):
        v = np.zeros(SHAPE, dtype=np.float32)
        v[head] = TISSUE["scalp"][i]
        v[skull_outer] = TISSUE["skull"][i]
        v[brain] = TISSUE["brain"][i]
        v[ventricles & brain] = TISSUE["csf"][i]
        v[edema] = TISSUE["edema"][i]
        v[enhancing] = TISSUE["enhance"][i]
        v[necrosis] = TISSUE["necrosis"][i]
        v = ndimage.gaussian_filter(v, 0.8)
        v += rng.normal(0, 0.02, SHAPE).astype(np.float32) * head
        vols[seq] = np.clip(v, 0, 1) * 1000
    return vols


def save_nifti(vol: np.ndarray, path: Path) -> None:
    affine = np.diag([SPACING, SPACING, SPACING, 1.0])
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), str(path))


def breast_ultrasound(rng: np.random.Generator) -> np.ndarray:
    h, w = 360, 480
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    tissue = 0.45 + 0.15 * np.sin(yy / 18) * np.cos(xx / 40)
    speckle = rng.rayleigh(0.25, (h, w)).astype(np.float32)
    img = tissue * speckle
    lesion = ((yy - 170) / 45) ** 2 + ((xx - 250) / 65) ** 2 <= 1
    img[lesion] *= 0.15
    shadow = (xx > 205) & (xx < 295) & (yy > 215)
    img[shadow] *= 0.6
    img[:20] = 0.9 * rng.random((20, w))  # skin line
    fan = np.abs(xx - w / 2) < (yy * 0.35 + 180)
    img[~fan] = 0
    return np.clip(ndimage.gaussian_filter(img, 1.0) * 255 / max(img.max(), 1e-6) * 1.6, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="demo_data")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    for tag, radius in (("t0", 9.0), ("t1", 12.0)):
        vols = phantom(radius, rng)
        for seq, vol in vols.items():
            save_nifti(vol, out / f"brain_{tag}_{seq}.nii.gz")
        if tag == "t0":
            # Axial slice through the tumor, displayed in radiological orientation.
            sl = vols["t1ce"][:, :, 46].T[::-1, ::-1]
            Image.fromarray((sl / sl.max() * 255).astype(np.uint8)).resize((256, 256)).save(out / "brain_slice.png")

    Image.fromarray(breast_ultrasound(rng)).save(out / "breast_us.png")
    print(f"Demo data written to {out.resolve()}")
    for p in sorted(out.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
