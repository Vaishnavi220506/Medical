"""Shared training utilities: devices, checkpoints, augmentation and metrics."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: torch.nn.Module, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def cosine_lr(optimizer, epoch: int, epochs: int, base_lr: float, warmup: int = 1) -> float:
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / (warmup + 1)
    else:
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# ---------------------------------------------------------------------- augmentation (GPU, batched)

def random_affine(x: torch.Tensor, y: torch.Tensor | None = None, max_rot: float = 12.0, scale=(0.9, 1.1), flip: bool = True):
    """Random rotation/scale/flip of a batch. ``y`` (label maps, B x H x W) gets the same transform."""
    b = x.shape[0]
    angle = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(b, device=x.device).uniform_(*scale)
    cos, sin = torch.cos(angle) / s, torch.sin(angle) / s
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    if flip:
        theta[:, 0, :2] *= torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)[:, None]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if y is None:
        return x
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    return x, y


def intensity_jitter(x: torch.Tensor) -> torch.Tensor:
    """Random gamma, contrast/brightness and noise on images in [0, 1]."""
    b = x.shape[0]
    shape = (b,) + (1,) * (x.ndim - 1)
    gamma = torch.empty(shape, device=x.device).uniform_(0.8, 1.25)
    x = x.clamp(0, 1) ** gamma
    x = x * torch.empty(shape, device=x.device).uniform_(0.9, 1.1) + torch.empty(shape, device=x.device).uniform_(-0.05, 0.05)
    x = x + torch.randn_like(x) * torch.empty(shape, device=x.device).uniform_(0, 0.03)
    return x.clamp(0, 1)


def sequence_dropout(x: torch.Tensor, p: float = 0.3, p_single: float = 0.15) -> torch.Tensor:
    """Simulate missing MRI sequences the way inference fills them (copy of an available one).

    With probability ``p_single`` every channel becomes the same sequence (a
    single uploaded image); with probability ``p`` a random subset is replaced.
    """
    if x.shape[1] == 1:
        return x
    x = x.clone()
    c = x.shape[1]
    for i in range(x.shape[0]):
        r = random.random()
        if r < p_single:
            src = random.randrange(c)
            x[i] = x[i, src : src + 1].expand_as(x[i])
        elif r < p_single + p:
            keep = random.randrange(c)
            for ch in range(c):
                if ch != keep and random.random() < 0.5:
                    x[i, ch] = x[i, keep]
    return x


# ---------------------------------------------------------------------- losses & metrics

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over foreground classes."""
    k = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, k).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice[1:].mean()


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.sum() + target.sum() == 0:
        return 1.0
    return float(2 * (pred & target).sum() / (pred.sum() + target.sum()))


def classification_metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1": round(float(f1), 4), "support": int(cm[i].sum())}
    return {
        "accuracy": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n": int(cm.sum()),
    }
