"""Train a tumor segmenter (2D U-Net).

Examples
--------
Brain MRI on BraTS 2021 (first converts the 3D cases into 2D slices once)::

    python -m oncovision.training.train_segmenter --profile brain_mri --format brats \\
        --data data/BraTS2021_Training_Data --prepared-dir data/brats_slices

Breast ultrasound on BUSI::

    python -m oncovision.training.train_segmenter --profile breast_us --format busi --data data/Dataset_BUSI_with_GT

Any images + masks folder (e.g. lung nodules)::

    python -m oncovision.training.train_segmenter --profile lung_ct --format maskfolder \\
        --data data/lung/images --masks data/lung/masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import settings
from ..models.networks import UNet
from ..models.registry import get_profile
from .common import cosine_lr, dice_loss, dice_score, intensity_jitter, pick_device, random_affine, save_checkpoint, seed_everything, sequence_dropout
from .datasets import (
    MaskSegmentationDataset,
    SliceFileDataset,
    find_busi_pairs,
    find_mask_folder_pairs,
    prepare_brats,
    split_by_case,
    split_by_key,
)


def evaluate(model, loader, device, profile, label_indices: list[int]) -> dict:
    """Mean per-image Dice: whole tumor / lesion, each label, and for BraTS models the standard regions."""
    model.eval()
    to_label = np.asarray(label_indices)
    multi = len(label_indices) > 2
    whole = "whole_tumor" if profile.organ == "brain" else "lesion"
    scores: dict[str, list[float]] = {}

    def add(name, pred, target):
        if pred.any() or target.any():  # empty-vs-empty would inflate Dice
            scores.setdefault(name, []).append(dice_score(pred, target))

    with torch.no_grad():
        for x, y in loader:
            pred = to_label[model(x.to(device)).argmax(1).cpu().numpy()]
            target = to_label[y.numpy()]
            for p, t in zip(pred, target):
                add(whole, p > 0, t > 0)
                if multi:
                    for lbl in profile.seg_labels:
                        add(lbl.name, p == lbl.index, t == lbl.index)
                    if profile.organ == "brain":
                        add("tumor_core", np.isin(p, (1, 3)), np.isin(t, (1, 3)))
                        add("enhancing_tumor_region", p == 3, t == 3)
    out = {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    main_keys = [k for k in (whole, "tumor_core", "enhancing_tumor_region") if k in out]
    out["mean_dice"] = round(float(np.mean([out[k] for k in main_keys])), 4) if main_keys else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--format", choices=("brats", "busi", "maskfolder"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, help="Mask folder for --format maskfolder")
    parser.add_argument("--prepared-dir", type=Path, help="Where BraTS slices are written (default: <data>_slices)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32, help="U-Net width (32 default; 16 for small GPUs)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()

    seed_everything()
    device = pick_device()
    profile = get_profile(args.profile)
    size = profile.seg_size
    out = args.out or settings.weights_dir / profile.key / "segmenter.pt"

    if args.format == "brats":
        prepared = args.prepared_dir or args.data.parent / f"{args.data.name}_slices"
        if not prepared.exists() or not any(prepared.glob("*.npz")):
            print(f"Preparing BraTS slices into {prepared} (one-time)...")
            prepare_brats(args.data, prepared, size)
        files = sorted(prepared.glob("*.npz"))
        if args.max_items:
            files = files[: args.max_items]
        train_files, val_files = split_by_case(files, args.val_fraction)
        train_ds, val_ds = SliceFileDataset(train_files), SliceFileDataset(val_files)
        channels = list(profile.seg_channels)
        label_indices = list(range(profile.num_seg_classes))
    else:
        # Binary masks: tumor vs background. For brain this is the whole-tumor label.
        pairs = find_busi_pairs(args.data) if args.format == "busi" else find_mask_folder_pairs(args.data, args.masks)
        if args.max_items:
            pairs = pairs[: args.max_items]
        # Files named "<patient>__<slice>.png" are split by patient.
        train_pairs, val_pairs = split_by_key(pairs, lambda pair: pair[0].stem.split("__")[0], args.val_fraction)
        train_ds, val_ds = MaskSegmentationDataset(train_pairs, size), MaskSegmentationDataset(val_pairs, size)
        channels = ["image"]
        label_indices = [0, profile.whole_tumor_label]
    if not len(train_ds):
        raise SystemExit("No training samples found. Check --data and --format (see docs/DATASETS.md).")
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  channels={channels}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
    model = UNet(in_channels=len(channels), num_classes=len(label_indices), base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    meta = {
        "profile": profile.key,
        "channels": channels,
        "num_classes": len(label_indices),
        "label_indices": label_indices,
        "image_size": size,
        "base": args.base,
        "architecture": "unet2d",
    }

    best = -1.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.lr)
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = random_affine(x, y)
            x = intensity_jitter(sequence_dropout(x))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = ce(logits, y) + dice_loss(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            seen += len(y)
        metrics = evaluate(model, val_loader, device, profile, label_indices) if len(val_ds) else {"mean_dice": 0.0}
        print(f"epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  loss={total / max(seen, 1):.4f}  val={json.dumps(metrics)}")
        if metrics["mean_dice"] >= best:
            best = metrics["mean_dice"]
            save_checkpoint(out, model, {**meta, "metrics": {"validation": metrics}, "epoch": epoch + 1})
    print(f"best mean Dice {best:.4f}; saved {out}")


if __name__ == "__main__":
    main()
