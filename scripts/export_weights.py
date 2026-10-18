"""Copy trained checkpoints into weights/ with float16 tensors (half the file size).

The engine loads them back into float32 models, so predictions are unchanged
for practical purposes.

Usage:  python scripts/export_weights.py --src runs/weights --dst weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, default=Path("weights"))
    args = parser.parse_args()
    for src in sorted(args.src.rglob("*.pt")):
        ckpt = torch.load(src, map_location="cpu", weights_only=True)
        state = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
        dst = args.dst / src.relative_to(args.src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "meta": {**ckpt["meta"], "stored_dtype": "float16"}}, dst)
        print(f"{src} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
