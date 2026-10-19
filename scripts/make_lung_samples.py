"""Write LIDC-IDRI CTs from the Hugging Face NIfTI mirror as correctly oriented NIfTI files.

The mirror's NIfTI headers swap the x and y axes, so viewers (and OncoVision) would show
the scan rotated. This writes each case with a fixed header, plus its nodule mask.

Usage:  python scripts/make_lung_samples.py --cases LIDC-IDRI-0001 --out samples/lung_ct_3d
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oncovision.imaging.io import load_nifti

GROUPS = ("malignant_cases_CT", "benign_cases_CT1", "benign_cases_CT2")


def fixed(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    scan = load_nifti(path.read_bytes())
    vol = np.ascontiguousarray(np.rot90(scan.channels["image"], 1, axes=(1, 2))[:, :, ::-1])
    dz, dy, dx = scan.spacing
    return vol, (dz, dx, dy)


def save(vol: np.ndarray, spacing, path: Path, dtype) -> None:
    """Inverse of load_nifti's reorientation: (Z, Y, X) display order -> RAS voxel order."""
    dz, dy, dx = spacing
    arr = np.ascontiguousarray(vol[:, ::-1, ::-1].transpose(2, 1, 0)).astype(dtype)
    img = nib.Nifti1Image(arr, np.diag([dx, dy, dz, 1.0]))
    img.header.set_xyzt_units("mm")
    nib.save(img, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Local copy of the mirror (otherwise downloaded)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    for case in args.cases:
        for group in GROUPS:
            mask_dir = "Mask" if group == "benign_cases_CT2" else "MASK"
            vol_rel, mask_rel = f"{group}/{case}_0000.nii.gz", f"{group}/{mask_dir}/{case}.nii.gz"
            if args.raw and (args.raw / vol_rel).exists():
                vol_path, mask_path = args.raw / vol_rel, args.raw / mask_rel
                break
            try:
                from huggingface_hub import hf_hub_download

                vol_path = Path(hf_hub_download("totalol/lungct", vol_rel, repo_type="dataset", local_dir=tmp))
                mask_path = Path(hf_hub_download("totalol/lungct", mask_rel, repo_type="dataset", local_dir=tmp))
                break
            except Exception:
                continue
        else:
            print(f"{case}: not found")
            continue
        vol, spacing = fixed(vol_path)
        mask, _ = fixed(mask_path)
        save(vol, spacing, args.out / f"{case}_ct.nii.gz", np.int16)
        save(mask > 0.5, spacing, args.out / f"{case}_nodule_mask.nii.gz", np.uint8)
        print(f"{case}: {vol.shape}, spacing {tuple(round(s, 2) for s in spacing)}, nodule voxels {int((mask > 0.5).sum())}")


if __name__ == "__main__":
    main()
