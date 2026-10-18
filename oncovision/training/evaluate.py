"""Evaluate a trained model and produce figures for a project report.

Classifier (writes metrics.json, confusion_matrix.png, gradcam_examples.png)::

    python -m oncovision.training.evaluate classifier --profile brain_mri --data data/Testing

Segmenter (writes metrics.json, segmentation_examples.png)::

    python -m oncovision.training.evaluate segmenter --profile breast_us --format busi --data data/Dataset_BUSI_with_GT
    python -m oncovision.training.evaluate segmenter --profile brain_mri --format brats --prepared-dir data/brats_slices

Output goes to ``reports/<profile>_<kind>/`` unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ..config import settings
from ..imaging.render import _HEAT_STOPS, blend_map, gray_rgb, overlay
from ..models.networks import Classifier, UNet, grad_cam
from ..models.registry import get_profile
from .common import classification_metrics, pick_device
from .datasets import (
    ClassificationDataset,
    MaskSegmentationDataset,
    SliceFileDataset,
    find_busi_pairs,
    find_class_images,
    find_mask_folder_pairs,
    split_by_case,
    split_by_key,
)
from .train_segmenter import evaluate as seg_evaluate

TILE = 192
BG = (17, 24, 39)
FG = (229, 231, 235)


def _load(path: Path, build, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build(ckpt["meta"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["meta"]


def _tile(rgb: np.ndarray, caption: str) -> Image.Image:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((TILE, TILE))
    canvas = Image.new("RGB", (TILE, TILE + 18), BG)
    canvas.paste(img, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), caption[:34], fill=FG)
    return canvas


def _grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    out = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (0, 0, 0))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    return out


def confusion_matrix_image(cm: list[list[int]], classes: list[str]) -> Image.Image:
    cm = np.array(cm)
    k, cell, left, top = len(classes), 90, 130, 60
    img = Image.new("RGB", (left + k * cell + 20, top + k * cell + 50), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((left, 10), "Predicted", fill=(0, 0, 0))
    draw.text((10, top + k * cell + 25), "rows = true class", fill=(80, 80, 80))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for j, name in enumerate(classes):
        draw.text((left + j * cell + 6, top - 20), name[:12], fill=(0, 0, 0))
    for i, name in enumerate(classes):
        draw.text((10, top + i * cell + cell // 2 - 6), name[:16], fill=(0, 0, 0))
        for j in range(k):
            shade = int(255 - 200 * norm[i, j])
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=(shade, shade + (255 - shade) // 3, 255))
            color = (255, 255, 255) if norm[i, j] > 0.55 else (0, 0, 0)
            draw.text((x0 + 8, y0 + cell // 2 - 12), f"{cm[i, j]}\n{norm[i, j]:.0%}", fill=color)
    return img


def eval_classifier(args, device) -> dict:
    profile = get_profile(args.profile)
    path = args.weights or settings.weights_dir / profile.key / "classifier.pt"
    model, meta = _load(path, lambda m: Classifier(num_classes=len(m["classes"])), device)
    classes, size = list(meta["classes"]), meta.get("image_size", 224)
    items = find_class_images(args.data, classes)
    if not items:
        raise SystemExit(f"No images for classes {classes} under {args.data}")
    loader = DataLoader(ClassificationDataset(items, size), batch_size=64, num_workers=args.workers)
    y_true, y_pred, conf = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1).cpu()
            y_true += y.tolist()
            y_pred += probs.argmax(1).tolist()
            conf += probs.max(1).values.tolist()
    metrics = classification_metrics(y_true, y_pred, classes)
    metrics["mean_confidence_correct"] = round(float(np.mean([c for c, t, p in zip(conf, y_true, y_pred) if t == p] or [0])), 4)
    metrics["mean_confidence_wrong"] = round(float(np.mean([c for c, t, p in zip(conf, y_true, y_pred) if t != p] or [0])), 4)

    confusion_matrix_image(metrics["confusion_matrix"], classes).save(args.out / "confusion_matrix.png")

    # Grad-CAM examples: a few correct and (if any) wrong predictions per class.
    rng = random.Random(0)
    picks = []
    for c in range(len(classes)):
        idx = [i for i, t in enumerate(y_true) if t == c]
        rng.shuffle(idx)
        picks += idx[:2]
    wrong = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    rng.shuffle(wrong)
    picks += wrong[:4]
    tiles = []
    for i in picks:
        img, _ = loader.dataset[i]
        cam = grad_cam(model, img[None].to(device), y_pred[i]).cpu().numpy()
        base = img[0].numpy()
        mark = "OK" if y_true[i] == y_pred[i] else "WRONG"
        tiles.append(_tile(gray_rgb(base), f"true: {classes[y_true[i]]}"))
        tiles.append(_tile(blend_map(base, cam, _HEAT_STOPS), f"{mark} pred: {classes[y_pred[i]]} {conf[i]:.0%}"))
    if tiles:
        _grid(tiles, 4).save(args.out / "gradcam_examples.png")
    return metrics


def eval_segmenter(args, device) -> dict:
    profile = get_profile(args.profile)
    path = args.weights or settings.weights_dir / profile.key / "segmenter.pt"
    model, meta = _load(path, lambda m: UNet(len(m["channels"]), m["num_classes"], base=m.get("base", 32)), device)
    size = meta.get("image_size", profile.seg_size)
    label_indices = meta.get("label_indices", list(range(meta["num_classes"])))
    to_label = np.asarray(label_indices)
    if args.format == "brats":
        files = sorted(Path(args.prepared_dir).glob("*.npz"))
        dataset = SliceFileDataset(split_by_case(files, args.val_fraction)[1] if args.validation_only else files)
    else:
        pairs = find_busi_pairs(args.data) if args.format == "busi" else find_mask_folder_pairs(args.data, args.masks)
        if args.validation_only:
            pairs = split_by_key(pairs, lambda pair: pair[0].stem.split("__")[0], args.val_fraction)[1]
        dataset = MaskSegmentationDataset(pairs, size)
    if not len(dataset):
        raise SystemExit("No samples found.")
    loader = DataLoader(dataset, batch_size=16, num_workers=args.workers)
    metrics = seg_evaluate(model, loader, device, profile, label_indices)
    metrics["n"] = len(dataset)

    # Examples: image | ground truth | prediction, preferring samples that contain tumor.
    with_tumor = [i for i in range(len(dataset)) if dataset[i][1].any()]
    rng = random.Random(0)
    rng.shuffle(with_tumor)
    tiles = []
    with torch.no_grad():
        for i in with_tumor[:6]:
            x, y = dataset[i]
            pred = to_label[model(x[None].to(device)).argmax(1)[0].cpu().numpy()]
            base = x[min(2, x.shape[0] - 1)].numpy()  # T1ce for BraTS
            truth = to_label[y.numpy()]
            d = 2 * ((pred > 0) & (truth > 0)).sum() / max((pred > 0).sum() + (truth > 0).sum(), 1)
            tiles += [
                _tile(gray_rgb(base), "image"),
                _tile(overlay(base, truth, profile), "ground truth"),
                _tile(overlay(base, pred, profile), f"prediction (Dice {d:.2f})"),
            ]
    if tiles:
        _grid(tiles, 3).save(args.out / "segmentation_examples.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("classifier", "segmenter"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data", type=Path, help="Test images (classifier) or dataset folder (segmenter)")
    parser.add_argument("--format", choices=("brats", "busi", "maskfolder"))
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--prepared-dir", type=Path, help="Prepared BraTS slices")
    parser.add_argument("--validation-only", action="store_true", help="Segmenter: evaluate only the validation split used in training")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.kind == "segmenter" and not args.format:
        parser.error("--format is required for segmenter evaluation")
    if args.kind == "classifier" and not args.data:
        parser.error("--data is required for classifier evaluation")

    args.out = args.out or Path("reports") / f"{args.profile}_{args.kind}"
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    metrics = eval_classifier(args, device) if args.kind == "classifier" else eval_segmenter(args, device)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"Report files written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
