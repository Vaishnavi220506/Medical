"""Train a 3D lung nodule segmenter (3D U-Net) on chest CT from LIDC-IDRI.

Each CT is clipped to the lung window, scaled to [0, 1] and resampled to 1 mm, exactly
as the app does, then cropped to the lungs. Training uses 3D patches, half of them
centred on a nodule. Validation reports voxel Dice and nodule-level detection
(sensitivity and false positives per scan). Needs a GPU (a free Kaggle T4 works);
see notebooks/train_lung3d.ipynb.

    python -m oncovision.training.train_lung3d --hf --prepared-dir /tmp/lung3d --epochs 60
    python -m oncovision.training.train_lung3d --data data/lidc --prepared-dir /tmp/lung3d

``--hf`` streams the NIfTI conversion of LIDC-IDRI from Hugging Face (totalol/lungct,
641 CTs, 38 GB) one case at a time, so the raw data never fills the disk.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from ..config import settings
from ..imaging.io import load_nifti
from ..imaging.lung import lung_box, lung_mask
from ..imaging.preprocess import normalize
from ..models.networks import UNet3D, sliding_window_3d
from ..models.registry import get_profile
from .common import cosine_lr, dice_loss, dice_score, pick_device, save_checkpoint, seed_everything

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
    x = to_model_space(vol, spacing, window)
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
    if workers > 1 and data is None:
        with ProcessPoolExecutor(workers) as pool:
            for n, msg in enumerate(pool.map(fn, jobs), 1):
                print(f"[{n}/{len(jobs)}] {msg}", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            print(f"[{n}/{len(jobs)}] {fn(j)}", flush=True)
    return sorted(out.glob("*.npz"))


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return d["image"], d["label"]


class PatchDataset(Dataset):
    """Random 3D patches from cases held in memory; half are centred on a nodule."""

    def __init__(self, cases: list[tuple[np.ndarray, np.ndarray]], patch: int, samples_per_epoch: int, fg_fraction: float = 0.5):
        self.cases, self.patch, self.n, self.fg = cases, patch, samples_per_epoch, fg_fraction
        self.fg_voxels = [np.argwhere(lab > 0) for _, lab in cases]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = random.randrange(len(self.cases))
        img, lab = self.cases[k]
        p = self.patch
        if random.random() < self.fg and len(self.fg_voxels[k]):
            center = self.fg_voxels[k][random.randrange(len(self.fg_voxels[k]))] + np.random.randint(-p // 4, p // 4 + 1, 3)
        else:
            center = [random.randrange(s) for s in lab.shape]
        start = [int(np.clip(c - p // 2, 0, max(s - p, 0))) for c, s in zip(center, lab.shape)]
        sl = tuple(slice(a, a + p) for a in start)
        x, y = img[sl].astype(np.float32) / 255.0, lab[sl].astype(np.int64)
        pad = [(0, p - s) for s in y.shape]
        x, y = np.pad(x, pad), np.pad(y, pad)
        for axis in range(3):
            if random.random() < 0.5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        if random.random() < 0.5:  # in-plane transpose: robust to rotated scan orientations
            x, y = x.transpose(0, 2, 1), y.transpose(0, 2, 1)
        x = np.clip(x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x[None])).float(), torch.from_numpy(np.ascontiguousarray(y))


def nodule_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    comp, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, comp, range(1, n + 1))
        small = np.where(sizes < MIN_NODULE_VOXELS)[0] + 1
        comp[np.isin(comp, small)] = 0
        comp, n = ndimage.label(comp > 0)
    return comp, n


def postprocess(prob: np.ndarray, lungs: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold, keep components at least 3 mm across that touch the (dilated) lungs."""
    pred = prob > threshold
    comp, n = nodule_components(pred)
    if n:
        inside = ndimage.maximum(lungs, comp, range(1, n + 1))
        pred = np.isin(comp, np.where(np.asarray(inside) > 0)[0] + 1)
    return pred


def evaluate(model, cases, device, patch: int, thresholds=(0.3, 0.5, 0.7)) -> dict:
    """Voxel Dice per scan with nodules, and nodule detection: sensitivity and false positives per scan."""
    out = {}
    probs, lungs = [], []
    for img, lab in cases:
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None].to(device)
        lungs.append(ndimage.binary_dilation(lung_mask(img.astype(np.float32) / 255.0), iterations=3))
        probs.append(sliding_window_3d(model, x, patch=patch, overlap=0.25, flip_tta=False)[1].cpu().numpy().astype(np.float16))
    for t in thresholds:
        dices, hits, total, fps = [], 0, 0, 0
        for (_, lab), prob, lung in zip(cases, probs, lungs):
            pred = postprocess(prob, lung, t)
            gt_comp, n_gt = nodule_components(lab > 0)
            pr_comp, n_pr = ndimage.label(pred)
            if n_gt:
                dices.append(dice_score(pred, lab > 0))
            # A predicted nodule is correct if at least 10% of it is a real nodule (so a
            # huge blob covering half the lung does not count as finding the nodule in it).
            good = np.zeros(n_pr + 1, bool)
            if n_pr:
                good[1:] = np.asarray(ndimage.mean(lab > 0, pr_comp, range(1, n_pr + 1))) >= 0.1
            total += n_gt
            hits += sum(bool(good[np.unique(pr_comp[gt_comp == i])].any()) for i in range(1, n_gt + 1))
            fps += int((~good[1:]).sum())
        out[str(t)] = {
            "dice": round(float(np.mean(dices)), 4) if dices else 0.0,
            "sensitivity": round(hits / max(total, 1), 4),
            "fp_per_scan": round(fps / max(len(cases), 1), 2),
            "nodules": total,
        }
    best_t = max(out, key=lambda k: out[k]["sensitivity"] - 0.05 * out[k]["fp_per_scan"] + out[k]["dice"])
    return {**out[best_t], "threshold": float(best_t), "mean_dice": out[best_t]["dice"], "by_threshold": out, "n_patients": len(cases)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", action="store_true", help=f"Stream LIDC-IDRI from Hugging Face ({HF_REPO})")
    src.add_argument("--data", type=Path, help="Local folder with <group>/<case>_0000.nii.gz and <group>/MASK/<case>.nii.gz")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=800)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-cases", type=int, default=40)
    parser.add_argument("--test-cases", type=int, default=60)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.patch % 16:
        parser.error("--patch must be divisible by 16")

    seed_everything()
    device = pick_device()
    files = prepare(args.prepared_dir, args.data, args.workers, args.max_cases)
    if args.max_cases:
        files = files[: args.max_cases]
    random.Random(42).shuffle(files)  # one CT per LIDC patient, so this is a patient-level split
    test, val, train = files[: args.test_cases], files[args.test_cases : args.test_cases + args.val_cases], files[args.test_cases + args.val_cases :]
    if not train or not val:
        raise SystemExit(f"Need more cases (found {len(files)}).")
    print(f"device={device}  train={len(train)}  val={len(val)}  test={len(test)} patients", flush=True)
    train_cases = [_load(f) for f in train]
    val_cases = [_load(f) for f in val]

    loader = DataLoader(PatchDataset(train_cases, args.patch, args.samples_per_epoch), batch_size=args.batch_size, num_workers=args.workers)
    model = UNet3D(1, 2, base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))  # nodules are tiny
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = args.out or settings.weights_dir / "lung_ct" / "segmenter3d.pt"
    meta = {
        "profile": "lung_ct",
        "architecture": "unet3d",
        "channels": ["image"],
        "num_classes": 2,
        "label_indices": [0, 1],
        "base": args.base,
        "patch": args.patch,
        "spacing_mm": SPACING,
        "crop": "lungs",
        "min_component_voxels": MIN_NODULE_VOXELS,
        "data": "LIDC-IDRI (totalol/lungct NIfTI conversion)",
        "val_cases": sorted(f.stem for f in val),
        "test_cases": sorted(f.stem for f in test),
    }
    best = -1.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.lr)
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = ce(logits, y) + dice_loss(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        msg = f"epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  loss={total / max(len(loader), 1):.4f}"
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            metrics = evaluate(model, val_cases, device, args.patch)
            score = metrics["sensitivity"] + metrics["dice"] - 0.05 * metrics["fp_per_scan"]
            msg += f"  val={json.dumps({k: v for k, v in metrics.items() if k != 'by_threshold'})}"
            if score >= best:
                best = score
                save_checkpoint(out, model, {**meta, "threshold": metrics["threshold"], "metrics": {"validation": metrics}, "epoch": epoch + 1})
        print(msg, flush=True)

    ckpt = torch.load(out, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    final = ckpt["meta"]["metrics"]
    if test:
        final["test"] = evaluate(model, [_load(f) for f in test], device, args.patch, thresholds=(ckpt["meta"]["threshold"],))
        print(f"test={json.dumps(final['test'])}")
    save_checkpoint(out, model, {**ckpt["meta"], "metrics": final})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
