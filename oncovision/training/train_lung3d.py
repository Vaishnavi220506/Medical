"""Train a 3D lung nodule segmenter (3D U-Net) on chest CT from LIDC-IDRI.

Each CT is clipped to the lung window, scaled to [0, 1] and resampled to 1 mm, exactly
as the app does, then cropped to the lungs. Training (oncovision.training.lesion3d) uses
3D patches, half of them centred on a nodule. Validation reports voxel Dice and
nodule-level detection (sensitivity and false positives per scan). Needs a GPU (a free Kaggle T4 works);
see notebooks/train_lung3d.ipynb.

    python -m oncovision.training.train_lung3d --hf --prepared-dir /tmp/lung3d --epochs 60
    python -m oncovision.training.train_lung3d --data data/lidc --prepared-dir /tmp/lung3d

``--hf`` streams the NIfTI conversion of LIDC-IDRI from Hugging Face (totalol/lungct,
641 CTs, 38 GB) one case at a time, so the raw data never fills the disk.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import settings
from ..imaging.io import load_nifti
from ..imaging.lung import lung_box, lung_mask
from ..imaging.preprocess import normalize
from ..models.registry import get_profile
from .lesion3d import add_training_args, train, write_splits

HF_REPO = "totalol/lungct"
SPACING = 1.0
MIN_NODULE_VOXELS = 14  # a 3 mm sphere; LIDC only marks nodules from 3 mm up


def to_model_space(vol_hu: np.ndarray, spacing, window) -> np.ndarray:
    """HU volume -> lung window, [0, 1] (like the app's channel_stack), 1 mm grid."""
    x = normalize(np.clip(vol_hu, *window))
    factors = [s / SPACING for s in spacing]
    return ndimage.zoom(x, factors, order=1) if any(abs(f - 1) > 0.05 for f in factors) else x


def prepare_case(vol_path: Path, mask_path: Path, target: Path, label: str, reorient: bool = True) -> str:
    window = get_profile("lung_ct").intensity_window
    scan = load_nifti(vol_path.read_bytes())
    vol, spacing = scan.channels["image"], scan.spacing
    mask = load_nifti(mask_path.read_bytes()).channels["image"] > 0.5
    if reorient:  # this mirror's NIfTI headers swap x and y
        vol = np.ascontiguousarray(np.rot90(vol, 1, axes=(1, 2))[:, :, ::-1])
        mask = np.ascontiguousarray(np.rot90(mask, 1, axes=(1, 2))[:, :, ::-1])
        spacing = (spacing[0], spacing[2], spacing[1])
    x = to_model_space(vol, spacing, window).astype(np.float32)
    del vol, scan
    m = ndimage.zoom(mask.astype(np.float32), [a / b for a, b in zip(x.shape, mask.shape)], order=1) > 0.5
    box = lung_box(x) or tuple(slice(0, s) for s in x.shape)
    np.savez_compressed(target, image=(x[box] * 255).astype(np.uint8), label=m[box].astype(np.uint8), malignant=label == "malignant")
    return f"{target.stem}: {x[box].shape}, nodule voxels {int(m.sum())}"


def _hf_job(args) -> str:
    vol_file, mask_file, target, label = args
    from huggingface_hub import hf_hub_download

    tmp = Path(tempfile.mkdtemp())
    try:
        v = hf_hub_download(HF_REPO, vol_file, repo_type="dataset", local_dir=tmp)
        m = hf_hub_download(HF_REPO, mask_file, repo_type="dataset", local_dir=tmp)
        return prepare_case(Path(v), Path(m), Path(target), label)
    except Exception as exc:  # one broken case should not stop the whole run
        return f"skip {Path(target).stem}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def prepare(out: Path, data: Path | None, workers: int, max_cases: int | None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if data is None:
        from huggingface_hub import list_repo_files

        files = set(list_repo_files(HF_REPO, repo_type="dataset"))
        for f in sorted(files):
            if f.endswith("_0000.nii.gz"):
                folder, name = f.rsplit("/", 1)
                case = name.removesuffix("_0000.nii.gz")
                # The mask folder is "MASK" in some groups and "Mask" in others.
                mask = next((m for d in ("MASK", "Mask", "mask") if (m := f"{folder}/{d}/{case}.nii.gz") in files), None)
                if mask:
                    jobs.append((f, mask, str(out / f"{case}.npz"), "malignant" if "malignant" in folder else "benign"))
    else:
        for v in sorted(Path(data).rglob("*_0000.nii.gz")):
            case = v.name.removesuffix("_0000.nii.gz")
            mask = next((m for d in ("MASK", "Mask", "mask") if (m := v.parent / d / f"{case}.nii.gz").exists()), None)
            if mask:
                jobs.append((v, mask, out / f"{case}.npz", "malignant" if "malignant" in v.parent.name else "benign"))
    jobs = [j for j in jobs[:max_cases] if not Path(j[2]).exists()]
    print(f"preparing {len(jobs)} cases", flush=True)
    fn = _hf_job if data is None else (lambda j: prepare_case(*j))
    done = 0
    if workers > 1 and data is None:
        try:
            with ProcessPoolExecutor(workers) as pool:
                for msg in pool.map(fn, jobs):
                    done += 1
                    print(f"[{done}/{len(jobs)}] {msg}", flush=True)
        except BrokenProcessPool:
            # A worker died, usually out of memory on a very long CT: finish one case at a time.
            print("A worker stopped (out of memory?); continuing one case at a time.", flush=True)
    for j in jobs:
        if not Path(j[2]).exists() and done < len(jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {fn(j)}", flush=True)
    return sorted(out.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", action="store_true", help=f"Stream LIDC-IDRI from Hugging Face ({HF_REPO})")
    src.add_argument("--data", type=Path, help="Local folder with <group>/<case>_0000.nii.gz and <group>/MASK/<case>.nii.gz")
    add_training_args(parser, val_cases=40, test_cases=60)
    parser.add_argument("--prep-workers", type=int, default=2, help="CTs converted in parallel (each needs up to ~3 GB of RAM)")
    args = parser.parse_args()

    files = prepare(args.prepared_dir, args.data, args.prep_workers, args.max_cases)
    write_splits(args.prepared_dir, args.val_cases, args.test_cases)
    if args.prepare_only:
        return
    meta = {
        "profile": "lung_ct",
        "spacing_mm": SPACING,
        "crop": "lungs",
        "min_component_mm3": MIN_NODULE_VOXELS,
        "data": "LIDC-IDRI (totalol/lungct NIfTI conversion)",
    }
    train(args, files, args.out or settings.weights_dir / "lung_ct" / "segmenter3d.pt", meta, lung_mask, MIN_NODULE_VOXELS)


if __name__ == "__main__":
    main()
