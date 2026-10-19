"""Cut 2D training slices out of prepared 3D cases (lung CT, breast MRI).

This gives the single-image (2D) models the same patients, and the same patient-level
train / test split, as the 3D model: only the 3D model's training patients are exported
with ``--split train``. Slices are axial, in the app's normalization, named
``<case>__<slice>.png`` so the 2D trainer also splits by patient.

    python -m oncovision.training.export_slices --prepared-dir /tmp/lung3d --out /tmp/lung2d --split train
    python -m oncovision.training.train_segmenter --profile lung_ct --format maskfolder \\
        --data /tmp/lung2d/images --masks /tmp/lung2d/masks

With ``--router N`` it also writes N plain slices (from any split) to ``<out>/router/`` for
teaching the scan-type router what this kind of scan looks like.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .lesion3d import load_case


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--lesion-slices", type=int, default=6, help="Slices through lesions per case")
    parser.add_argument("--empty-slices", type=int, default=3, help="Lesion-free slices per case")
    parser.add_argument("--router", type=int, default=0, help="Also write this many slices for the scan-type router")
    args = parser.parse_args()
    rng = random.Random(0)

    splits = json.loads((args.prepared_dir / "splits.json").read_text())
    names = sorted({n for k, v in splits.items() for n in v if args.split in ("all", k)})
    counts = {"lesion": 0, "empty": 0}
    router = []
    for name in names:
        img, lab = load_case(args.prepared_dir / f"{name}.npz")
        has = lab.any(axis=(1, 2))
        body = (img > 5).mean(axis=(1, 2))
        lesion_z = list(np.where(has)[0])
        far = [z for z in range(img.shape[0]) if body[z] > 0.15 and (not lesion_z or min(abs(z - w) for w in lesion_z) >= 8)]
        picks = [(z, True) for z in rng.sample(lesion_z, min(args.lesion_slices, len(lesion_z)))]
        picks += [(z, False) for z in rng.sample(far, min(args.empty_slices, len(far)))]
        for z, lesion in picks:
            stem = f"{name}__{z:03d}.png"
            _save(img[z], args.out / "images" / stem)
            _save(lab[z].astype(np.uint8) * 255, args.out / "masks" / stem)
            counts["lesion" if lesion else "empty"] += 1
        if args.router:
            router += [(name, img[z]) for z in rng.sample(range(img.shape[0]), min(3, img.shape[0])) if body[z] > 0.15]
    rng.shuffle(router)
    for name, sl in router[: args.router]:
        _save(sl, args.out / "router" / f"{name}_{rng.randrange(10**6)}.png")
    print(f"{len(names)} cases -> {counts['lesion']} lesion slices, {counts['empty']} empty slices, {min(len(router), args.router)} router slices")


if __name__ == "__main__":
    main()
