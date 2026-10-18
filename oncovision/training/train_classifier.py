"""Train a tumor classifier (or the scan-type router).

Examples
--------
Brain MRI (Kaggle "Brain Tumor MRI Dataset", folders Training/ and Testing/)::

    python -m oncovision.training.train_classifier --profile brain_mri \\
        --train-dir data/brain-tumor-mri/Training --test-dir data/brain-tumor-mri/Testing --pretrained

Breast ultrasound (BUSI, folders benign/ malignant/ normal/)::

    python -m oncovision.training.train_classifier --profile breast_us --train-dir data/Dataset_BUSI_with_GT

Router (one folder per scan type: brain_mri/ breast_us/ lung_ct/ other/)::

    python -m oncovision.training.train_classifier --router --train-dir data/router
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import settings
from ..models.networks import Classifier
from ..models.registry import ROUTER_CLASSES, get_profile
from .common import classification_metrics, cosine_lr, intensity_jitter, pick_device, random_affine, save_checkpoint, seed_everything
from .datasets import ClassificationDataset, find_class_images, split_items


def evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_true += y.tolist()
            y_pred += logits.argmax(1).cpu().tolist()
    return y_true, y_pred


def train(
    classes: list[str],
    train_dir: Path,
    out: Path,
    test_dir: Path | None = None,
    image_size: int = 224,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    val_fraction: float = 0.15,
    pretrained: bool = False,
    workers: int = 2,
    max_items: int | None = None,
    extra_meta: dict | None = None,
) -> dict:
    seed_everything()
    device = pick_device()
    items = find_class_images(train_dir, classes)
    if not items:
        raise SystemExit(f"No images found in {train_dir} for classes {classes}. Check the folder names (see docs/DATASETS.md).")
    if max_items:
        items = items[:max_items]
    train_items, val_items = split_items(items, val_fraction)
    counts = Counter(label for _, label in train_items)
    print(f"device={device}  train={len(train_items)}  val={len(val_items)}  per class={ {classes[k]: v for k, v in sorted(counts.items())} }")

    train_loader = DataLoader(ClassificationDataset(train_items, image_size), batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=len(train_items) > batch_size)
    val_loader = DataLoader(ClassificationDataset(val_items, image_size), batch_size=batch_size, num_workers=workers)

    model = Classifier(num_classes=len(classes), pretrained=pretrained).to(device)
    # Inverse-frequency class weights handle imbalanced datasets such as BUSI.
    weights = torch.tensor([len(train_items) / (len(classes) * max(counts.get(i, 0), 1)) for i in range(len(classes))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    meta = {"classes": classes, "image_size": image_size, "architecture": "resnet18", **(extra_meta or {})}
    best = -1.0
    for epoch in range(epochs):
        current_lr = cosine_lr(optimizer, epoch, epochs, lr)
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = intensity_jitter(random_affine(x))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            seen += len(y)
        metrics = classification_metrics(*evaluate(model, val_loader, device), classes) if val_items else {"macro_f1": 0.0, "accuracy": 0.0}
        print(f"epoch {epoch + 1}/{epochs}  lr={current_lr:.2e}  loss={total / max(seen, 1):.4f}  val_acc={metrics['accuracy']:.4f}  val_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] >= best:
            best = metrics["macro_f1"]
            save_checkpoint(out, model, {**meta, "metrics": {"validation": metrics}, "epoch": epoch + 1})

    # Reload the best checkpoint and evaluate it on the held-out test set.
    ckpt = torch.load(out, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    final = ckpt["meta"]["metrics"]
    if test_dir:
        test_items = find_class_images(test_dir, classes)
        test_loader = DataLoader(ClassificationDataset(test_items, image_size), batch_size=batch_size, num_workers=workers)
        final["test"] = classification_metrics(*evaluate(model, test_loader, device), classes)
        print(f"test accuracy={final['test']['accuracy']:.4f}  macro F1={final['test']['macro_f1']:.4f}")
    save_checkpoint(out, model, {**ckpt["meta"], "metrics": final})
    print(json.dumps(final, indent=1))
    print(f"saved {out}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--profile", help="Scan type to train, e.g. brain_mri, breast_us, lung_ct")
    target.add_argument("--router", action="store_true", help="Train the scan-type router instead")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--out", type=Path, help="Checkpoint path (defaults to weights/...)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--pretrained", action="store_true", help="Start from ImageNet weights (downloads once)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-items", type=int, help="Use only the first N images (quick experiments)")
    args = parser.parse_args()

    if args.router:
        classes, size, out = list(ROUTER_CLASSES), 224, args.out or settings.weights_dir / "router.pt"
        extra = {"router": True}
    else:
        profile = get_profile(args.profile)
        classes, size = list(profile.classes), profile.image_size
        out = args.out or settings.weights_dir / profile.key / "classifier.pt"
        extra = {"profile": profile.key}
    train(classes, args.train_dir, out, args.test_dir, size, args.epochs, args.batch_size, args.lr, args.val_fraction, args.pretrained, args.workers, args.max_items, extra)


if __name__ == "__main__":
    main()
