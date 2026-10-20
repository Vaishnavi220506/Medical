"""Shared 3D lesion-segmentation training: patches, splits, detection metrics and the loop.

Used by train_lung3d (lung nodules on CT) and train_breast_mri3d (breast tumors on
DCE-MRI). Each prepared case is an ``.npz`` with ``image`` (uint8, model space: the
same normalization, voxel size and crop the app uses) and ``label`` (0/1).
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from ..models.networks import UNet3D, sliding_window_3d
from .common import cosine_lr, dice_loss, dice_score, pick_device, save_checkpoint, seed_everything


def load_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return d["image"], d["label"]


def split_cases(files: list[Path], val_cases: int, test_cases: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Fixed patient-level split (one scan per patient), shared by every model trained on these cases."""
    files = sorted(files)
    random.Random(42).shuffle(files)
    test, val = files[:test_cases], files[test_cases : test_cases + val_cases]
    return files[test_cases + val_cases :], val, test


def write_splits(prepared: Path, val_cases: int, test_cases: int) -> dict:
    train, val, test = split_cases(list(prepared.glob("*.npz")), val_cases, test_cases)
    splits = {"train": sorted(f.stem for f in train), "val": sorted(f.stem for f in val), "test": sorted(f.stem for f in test)}
    (prepared / "splits.json").write_text(json.dumps(splits))
    return splits


def to_npy(files: list[Path]) -> list[tuple[Path, np.ndarray]]:
    """Training cases as (uncompressed image on disk, lesion voxel positions in memory).

    Images are read memory-mapped during training, so hundreds of CTs never have to fit in RAM
    at once; lesions are small, so their voxel positions replace full-size label volumes.
    """
    out = []
    for f in files:
        img = f.with_name(f.stem + "_img.npy")
        image, label = load_case(f)
        if not img.exists():
            np.save(img, image)
        out.append((img, np.argwhere(label > 0).astype(np.int16)))
    return out


class PatchDataset(Dataset):
    """Random 3D patches read from memory-mapped images; half are centred on a lesion."""

    def __init__(self, cases: list[tuple[Path, np.ndarray]], patch: int, samples_per_epoch: int, fg_fraction: float = 0.5):
        self.cases, self.patch, self.n, self.fg = cases, patch, samples_per_epoch, fg_fraction
        self.fg_voxels = [fg for _, fg in cases]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = random.randrange(len(self.cases))
        img = np.load(self.cases[k][0], mmap_mode="r")
        fg = self.fg_voxels[k]
        p = self.patch
        if random.random() < self.fg and len(fg):
            center = fg[random.randrange(len(fg))] + np.random.randint(-p // 4, p // 4 + 1, 3)
        else:
            center = [random.randrange(s) for s in img.shape]
        start = [int(np.clip(c - p // 2, 0, max(s - p, 0))) for c, s in zip(center, img.shape)]
        sl = tuple(slice(a, a + p) for a in start)
        x = img[sl].astype(np.float32) / 255.0
        y = np.zeros(x.shape, np.int64)
        inside = np.all((fg >= start) & (fg < np.array(start) + np.array(x.shape)), axis=1)
        y[tuple((fg[inside] - start).T)] = 1
        pad = [(0, p - s) for s in y.shape]
        x, y = np.pad(x, pad), np.pad(y, pad)
        for axis in range(3):
            if random.random() < 0.5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        if random.random() < 0.5:  # in-plane transpose: robust to rotated scan orientations
            x, y = x.transpose(0, 2, 1), y.transpose(0, 2, 1)
        x = np.clip(x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05), 0, 1) ** np.random.uniform(0.85, 1.2)
        return torch.from_numpy(np.ascontiguousarray(x[None])).float(), torch.from_numpy(np.ascontiguousarray(y))


def lesion_components(mask: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    """Connected lesions, ignoring ones smaller than ``min_voxels``."""
    comp, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, comp, range(1, n + 1))
        comp[np.isin(comp, np.where(sizes < min_voxels)[0] + 1)] = 0
        comp, n = ndimage.label(comp > 0)
    return comp, n


def postprocess(prob: np.ndarray, region: np.ndarray | None, threshold: float, min_voxels: int) -> np.ndarray:
    """Threshold, drop findings under the minimum size and (if given) outside the organ region."""
    comp, n = lesion_components(prob > threshold, min_voxels)
    if n and region is not None:
        inside = np.asarray(ndimage.maximum(region, comp, range(1, n + 1))) > 0
        return np.isin(comp, np.where(inside)[0] + 1)
    return comp > 0


def evaluate(model, cases, regions, device, patch: int, min_voxels: int, thresholds=(0.3, 0.5, 0.7)) -> dict:
    """Voxel Dice per scan with lesions, and lesion detection: sensitivity and false positives per scan.

    A predicted lesion counts as correct if at least 10% of it is a real lesion, so a huge
    blob covering half the organ does not count as finding the lesion inside it.
    """
    probs = []
    for img, _ in cases:
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None].to(device)
        probs.append(sliding_window_3d(model, x, patch=patch, overlap=0.25, flip_tta=False)[1].cpu().numpy().astype(np.float16))
    out = {}
    for t in thresholds:
        dices, hits, total, fps = [], 0, 0, 0
        for (_, lab), prob, region in zip(cases, probs, regions):
            pred = postprocess(prob, region, t, min_voxels)
            gt_comp, n_gt = lesion_components(lab > 0, min_voxels)
            pr_comp, n_pr = ndimage.label(pred)
            if n_gt:
                dices.append(dice_score(pred, lab > 0))
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
            "lesions": total,
        }
    best_t = max(out, key=lambda k: out[k]["sensitivity"] - 0.05 * out[k]["fp_per_scan"] + out[k]["dice"])
    return {**out[best_t], "threshold": float(best_t), "mean_dice": out[best_t]["dice"], "by_threshold": out, "n_patients": len(cases)}


def add_training_args(parser: argparse.ArgumentParser, val_cases: int, test_cases: int) -> None:
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=800)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-cases", type=int, default=val_cases)
    parser.add_argument("--test-cases", type=int, default=test_cases)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true", help="Only convert the data (and write splits.json)")


def train(args, files: list[Path], out: Path, meta: dict, region_fn: Callable[[np.ndarray], np.ndarray] | None, min_voxels: int) -> None:
    """Train, keep the best validation checkpoint, then score it once on the test patients."""
    if args.patch % 16:
        raise SystemExit("--patch must be divisible by 16")
    seed_everything()
    device = pick_device()
    if args.max_cases:
        files = sorted(files)[: args.max_cases]
    train_files, val, test = split_cases(files, args.val_cases, args.test_cases)
    if not train_files or not val:
        raise SystemExit(f"Need more cases (found {len(files)}).")
    print(f"device={device}  train={len(train_files)}  val={len(val)}  test={len(test)} patients", flush=True)
    train_cases = to_npy(train_files)
    val_cases = [load_case(f) for f in val]

    def regions(cases):
        if region_fn is None:
            return [None] * len(cases)
        return [ndimage.binary_dilation(region_fn(img.astype(np.float32) / 255.0), iterations=3) for img, _ in cases]

    val_regions = regions(val_cases)
    loader = DataLoader(PatchDataset(train_cases, args.patch, args.samples_per_epoch), batch_size=args.batch_size, num_workers=args.workers)
    model = UNet3D(1, 2, base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))  # lesions are small
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    meta = {
        **meta,
        "architecture": "unet3d",
        "channels": ["image"],
        "num_classes": 2,
        "label_indices": [0, 1],
        "base": args.base,
        "patch": args.patch,
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
            metrics = evaluate(model, val_cases, val_regions, device, args.patch, min_voxels)
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
        del loader, train_cases  # free memory before loading the test patients
        test_cases = [load_case(f) for f in test]
        final["test"] = evaluate(model, test_cases, regions(test_cases), device, args.patch, min_voxels, thresholds=(ckpt["meta"]["threshold"],))
        print(f"test={json.dumps(final['test'])}")
    save_checkpoint(out, model, {**ckpt["meta"], "metrics": final})
    print(f"saved {out}")
