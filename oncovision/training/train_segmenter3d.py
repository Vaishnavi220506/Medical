"""Train a true 3D tumor segmenter (3D U-Net) on BraTS multi-sequence MRI.

It sees the whole brain in 3D with all four sequences (FLAIR, T1, T1ce, T2) and
outlines necrotic core, edema and enhancing tumor. Needs a GPU (a free Kaggle
T4 works); see notebooks/train_3d_brats.ipynb.

    python -m oncovision.training.train_segmenter3d --data data/BraTS2023 --prepared-dir /tmp/brats3d \\
        --epochs 40 --patch 128 --batch-size 2

BraTS 2021 and 2023 folder layouts and file names are both accepted.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import settings
from ..imaging.io import load_nifti
from ..imaging.preprocess import normalize
from ..models.networks import UNet3D, sliding_window_3d
from .common import cosine_lr, dice_loss, dice_score, pick_device, save_checkpoint, seed_everything, sequence_dropout
from .datasets import BRATS_SEQUENCES, _brats_file, patient_id

ALIASES = {"flair": ("flair", "t2f"), "t1": ("t1", "t1n"), "t1ce": ("t1ce", "t1c"), "t2": ("t2", "t2w")}
MARGIN = 4


def prepare(root: Path, out: Path) -> list[Path]:
    """One compressed file per case: uint8 sequences cropped to the brain, plus labels (1 NCR, 2 ED, 3 ET)."""
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(p for p in Path(root).rglob("*") if p.is_dir() and _brats_file(p, "seg") is not None)
    for n, case in enumerate(cases, 1):
        target = out / f"{case.name}.npz"
        if target.exists():
            continue
        paths = {s: next((f for a in ALIASES[s] if (f := _brats_file(case, a)) is not None), None) for s in BRATS_SEQUENCES}
        if any(v is None for v in paths.values()):
            continue
        vols = np.stack([normalize(load_nifti(paths[s].read_bytes()).channels["image"]) for s in BRATS_SEQUENCES])
        seg = load_nifti(_brats_file(case, "seg").read_bytes()).channels["image"].round().astype(np.uint8)
        seg[seg == 4] = 3
        brain = vols.max(0) > 0.02
        zz, yy, xx = (np.where(brain.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1)))
        sl = tuple(slice(max(a[0] - MARGIN, 0), a[-1] + MARGIN + 1) for a in (zz, yy, xx))
        np.savez_compressed(target, image=(vols[(slice(None), *sl)] * 255).astype(np.uint8), label=seg[sl])
        if n % 50 == 0:
            print(f"prepared {n}/{len(cases)} cases", flush=True)
    return sorted(out.glob("*.npz"))


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return d["image"].astype(np.float32) / 255.0, d["label"].astype(np.int64)


class PatchDataset(Dataset):
    """Random 3D patches; half are centred on a tumor voxel so the model sees enough tumor."""

    def __init__(self, files: list[Path], patch: int, samples_per_epoch: int, fg_fraction: float = 0.5):
        self.files, self.patch, self.n, self.fg = files, patch, samples_per_epoch, fg_fraction

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, lab = _load(random.choice(self.files))
        p = self.patch
        pad = [(0, max(0, p - s)) for s in lab.shape]
        img = np.pad(img, [(0, 0), *pad])
        lab = np.pad(lab, pad)
        if random.random() < self.fg and lab.any():
            center = np.argwhere(lab > 0)[random.randrange(int((lab > 0).sum()))]
        else:
            center = [random.randrange(s) for s in lab.shape]
        start = [int(np.clip(c - p // 2, 0, s - p)) for c, s in zip(center, lab.shape)]
        sl = tuple(slice(a, a + p) for a in start)
        x, y = img[(slice(None), *sl)], lab[sl]
        for axis in range(3):  # random flips
            if random.random() < 0.5:
                x, y = np.flip(x, axis + 1), np.flip(y, axis)
        x = x * np.random.uniform(0.9, 1.1, (x.shape[0], 1, 1, 1)) + np.random.uniform(-0.05, 0.05, (x.shape[0], 1, 1, 1))
        x = np.clip(x, 0, 1) ** np.random.uniform(0.8, 1.25)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y))


def evaluate(model, files: list[Path], device, patch: int) -> dict:
    """Mean per-patient Dice for the BraTS regions on whole volumes."""
    scores: dict[str, list[float]] = {"whole_tumor": [], "tumor_core": [], "enhancing_tumor": []}
    for f in files:
        img, lab = _load(f)
        probs = sliding_window_3d(model, torch.from_numpy(img).to(device), patch=patch, overlap=0.25, flip_tta=False)
        pred = probs.argmax(0).cpu().numpy()
        scores["whole_tumor"].append(dice_score(pred > 0, lab > 0))
        scores["tumor_core"].append(dice_score(np.isin(pred, (1, 3)), np.isin(lab, (1, 3))))
        scores["enhancing_tumor"].append(dice_score(pred == 3, lab == 3))
    out = {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    out["mean_dice"] = round(float(np.mean(list(out.values()))), 4)
    out["n_patients"] = len(files)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="BraTS training data folder (one folder per case)")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=500)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-cases", type=int, default=40, help="Held-out patients for validation")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--test-cases", type=int, default=0, help="Further held-out patients for a final test score")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.patch % 16:
        parser.error("--patch must be divisible by 16")

    seed_everything()
    device = pick_device()
    files = prepare(args.data, args.prepared_dir)
    if args.max_cases:
        files = files[: args.max_cases]
    # Split by patient (BraTS 2023 has several time points per patient).
    patients = sorted({patient_id(f.stem) for f in files})
    random.Random(42).shuffle(patients)
    test_p = set(patients[: args.test_cases])
    val_p = set(patients[args.test_cases : args.test_cases + args.val_cases])
    test = [f for f in files if patient_id(f.stem) in test_p]
    val = [f for f in files if patient_id(f.stem) in val_p]
    train = [f for f in files if patient_id(f.stem) not in test_p | val_p]
    if not train or not val:
        raise SystemExit(f"Need more cases (found {len(files)}).")
    print(f"device={device}  train={len(train)}  val={len(val)}  test={len(test)} patients", flush=True)

    loader = DataLoader(PatchDataset(train, args.patch, args.samples_per_epoch), batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    model = UNet3D(len(BRATS_SEQUENCES), 4, base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = args.out or settings.weights_dir / "brain_mri" / "segmenter3d.pt"
    meta = {
        "profile": "brain_mri",
        "architecture": "unet3d",
        "channels": list(BRATS_SEQUENCES),
        "num_classes": 4,
        "label_indices": [0, 1, 2, 3],
        "base": args.base,
        "patch": args.patch,
        "spacing_mm": 1.0,
    }
    best = -1.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.lr)
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = sequence_dropout(x.to(device)), y.to(device)
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
            metrics = evaluate(model, val, device, args.patch)
            msg += f"  val={json.dumps(metrics)}"
            if metrics["mean_dice"] >= best:
                best = metrics["mean_dice"]
                save_checkpoint(out, model, {**meta, "metrics": {"validation": metrics}, "epoch": epoch + 1})
        print(msg, flush=True)

    ckpt = torch.load(out, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    final = ckpt["meta"]["metrics"]
    if test:
        final["test"] = evaluate(model, test, device, args.patch)
        print(f"test={json.dumps(final['test'])}")
    save_checkpoint(out, model, {**ckpt["meta"], "metrics": final})
    print(f"best validation mean Dice {best:.4f}; saved {out}")


if __name__ == "__main__":
    main()
