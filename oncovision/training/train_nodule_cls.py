"""Train a 3D lung nodule classifier (benign vs malignant) on LIDC-IDRI nodules.

Uses the cases prepared by train_lung3d (1 mm, lung window) and the same patient split.
Each expert-outlined nodule is cut out as a 48 mm cube. The label is the case's group in
the source mirror (benign or malignant case), so it is a weak, case-level label: treat the
result as experimental. Scores are per patient (the most suspicious nodule), on held-out
test patients.

    python -m oncovision.training.train_nodule_cls --prepared-dir /tmp/lung3d --epochs 40
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from ..config import settings
from ..models.networks import NoduleClassifier3D
from .common import cosine_lr, pick_device, save_checkpoint, seed_everything
from .lesion3d import lesion_components

PATCH = 48
MIN_VOXELS = 14


def nodule_patch(img: np.ndarray, center, patch: int = PATCH) -> np.ndarray:
    """A cube around ``center`` (zero-padded at the edges), as float in [0, 1]."""
    half = patch // 2
    c = [int(round(v)) for v in center]
    out = np.zeros((patch,) * 3, np.float32)
    src = tuple(slice(max(ci - half, 0), min(ci + half, s)) for ci, s in zip(c, img.shape))
    dst = tuple(slice(sl.start - (ci - half), sl.stop - (ci - half)) for sl, ci in zip(src, c))
    out[dst] = img[src]
    return out


def collect(prepared: Path, names: list[str]) -> list[tuple[str, np.ndarray, int]]:
    """(case, patch, label) for every outlined nodule of these cases."""
    items = []
    for name in names:
        with np.load(prepared / f"{name}.npz") as d:
            img, lab, malignant = d["image"].astype(np.float32) / 255.0, d["label"] > 0, int(d["malignant"])
        comp, n = lesion_components(lab, MIN_VOXELS)
        for center in ndimage.center_of_mass(lab, comp, range(1, n + 1)) if n else []:
            items.append((name, nodule_patch(img, center), malignant))
    return items


class PatchSet(Dataset):
    def __init__(self, items, augment: bool):
        self.items, self.augment = items, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, x, y = self.items[i]
        if self.augment:
            shift = np.random.randint(-3, 4, 3)
            x = np.roll(x, shift, axis=(0, 1, 2))
            for axis in range(3):
                if random.random() < 0.5:
                    x = np.flip(x, axis)
            if random.random() < 0.5:
                x = x.transpose(0, 2, 1)
            x = np.clip(x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x[None])).float(), y


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def patient_scores(model, items, device) -> tuple[list[float], list[int], list[str]]:
    """Malignancy probability per patient: the most suspicious of their nodules."""
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(items), 32):
            x = torch.from_numpy(np.stack([p for _, p, _ in items[start : start + 32]])[:, None]).to(device)
            probs += torch.softmax(model(x), 1)[:, 1].cpu().tolist()
    best: dict[str, tuple[float, int]] = {}
    for (name, _, y), p in zip(items, probs):
        best[name] = (max(p, best.get(name, (0.0, y))[0]), y)
    names = sorted(best)
    return [best[n][0] for n in names], [best[n][1] for n in names], names


def metrics(scores, labels, threshold: float) -> dict:
    s, y = np.array(scores), np.array(labels)
    pred = s >= threshold
    return {
        "auc": round(auc(scores, labels), 4),
        "sensitivity": round(float(pred[y == 1].mean()), 4) if (y == 1).any() else None,
        "specificity": round(float((~pred[y == 0]).mean()), 4) if (y == 0).any() else None,
        "threshold": round(threshold, 4),
        "patients": len(y),
        "malignant_patients": int(y.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    seed_everything()
    device = pick_device()
    splits = json.loads((args.prepared_dir / "splits.json").read_text())
    train_items, val_items, test_items = (collect(args.prepared_dir, splits[k]) for k in ("train", "val", "test"))
    n_mal = sum(y for *_, y in train_items)
    print(f"device={device}  nodules: train={len(train_items)} ({n_mal} from malignant cases)  val={len(val_items)}  test={len(test_items)}", flush=True)

    # Balance the classes by sampling malignant nodules more often.
    weights = [1.0 / max(n_mal, 1) if y else 1.0 / max(len(train_items) - n_mal, 1) for *_, y in train_items]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
    loader = DataLoader(PatchSet(train_items, augment=True), batch_size=args.batch_size, sampler=sampler, num_workers=2)
    model = NoduleClassifier3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    out = args.out or settings.weights_dir / "lung_ct" / "nodule_classifier.pt"
    meta = {"profile": "lung_ct", "architecture": "nodule_cnn3d", "classes": ["benign", "malignant"], "patch": PATCH, "spacing_mm": 1.0,
            "label_source": "case-level benign/malignant group of the LIDC-IDRI mirror (weak label)"}
    best = -1.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.lr)
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = ce(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scores, labels, _ = patient_scores(model, val_items, device)
        val_auc = auc(scores, labels)
        print(f"epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  loss={total / max(len(loader), 1):.4f}  val_auc={val_auc:.4f}", flush=True)
        if val_auc >= best:
            best = val_auc
            # Threshold: best balance of sensitivity and specificity (Youden) on validation patients.
            s, y = np.array(scores), np.array(labels)
            threshold = float(max(np.unique(s), key=lambda t: (s[y == 1] >= t).mean() + (s[y == 0] < t).mean()))
            save_checkpoint(out, model, {**meta, "threshold": threshold, "metrics": {"validation": metrics(scores, labels, threshold)}, "epoch": epoch + 1})

    ckpt = torch.load(out, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    final = ckpt["meta"]["metrics"]
    scores, labels, _ = patient_scores(model, test_items, device)
    final["test"] = metrics(scores, labels, ckpt["meta"]["threshold"])
    print(f"test={json.dumps(final['test'])}")
    save_checkpoint(out, model, {**ckpt["meta"], "metrics": final})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
