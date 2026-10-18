"""Convert the Figshare brain tumor dataset (Cheng et al. 2015) into image/mask folders.

Source: https://huggingface.co/datasets/MedOtter/Figshare_Brain_Tumor (parquet, CC-BY-4.0),
3,064 T1-contrast slices from 233 patients with expert tumor masks.

Writes::

    <out>/train/images/<patient>__<slice>.png   (16-bit grayscale)
    <out>/train/masks/<patient>__<slice>.png    (0 / 255)
    <out>/test/...                              (fold 1, patient-disjoint from train)

The "<patient>__" prefix lets the trainer split train/validation by patient.

Usage:
    pip install pyarrow huggingface_hub
    python scripts/prepare_figshare.py --download --out data/figshare --negatives data/brain/Training/notumor
    python -m oncovision.training.train_segmenter --profile brain_mri --format maskfolder \\
        --data data/figshare/train/images --masks data/figshare/train/masks
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/figshare_raw"), help="Folder with the downloaded parquet files")
    parser.add_argument("--out", type=Path, default=Path("data/figshare"))
    parser.add_argument("--download", action="store_true", help="Download the dataset from Hugging Face first")
    parser.add_argument(
        "--negatives",
        type=Path,
        help="Folder of tumor-free brain MRIs (e.g. data/brain/Training/notumor). They are added to train/ with "
        "empty masks so the segmenter learns that a scan can contain no tumor. Strongly recommended.",
    )
    parser.add_argument("--max-negatives", type=int, default=800)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id="MedOtter/Figshare_Brain_Tumor", repo_type="dataset", local_dir=str(args.raw))

    files = sorted(args.raw.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files under {args.raw}. Use --download or point --raw at them.")
    counts: dict[str, int] = {}
    for f in files:
        split = "test" if f.name.startswith("test") else "train"
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "masks").mkdir(parents=True, exist_ok=True)
        table = pq.read_table(f, columns=["id", "pid", "label_name", "image", "mask"])
        for row in table.to_pylist():
            name = f"{row['pid']}__{row['id']}_{row['label_name']}.png"
            (args.out / split / "images" / name).write_bytes(row["image"]["bytes"])
            (args.out / split / "masks" / name).write_bytes(row["mask"]["bytes"])
            counts[split] = counts.get(split, 0) + 1
        print(f"{f.name}: done")
    if args.negatives:
        from PIL import Image

        files = sorted(p for p in args.negatives.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.max_negatives]
        for f in files:
            img = Image.open(f).convert("L")
            name = f"neg-{f.stem}__0.png"  # each negative is its own "patient" group
            img.save(args.out / "train" / "images" / name)
            Image.new("L", img.size, 0).save(args.out / "train" / "masks" / name)
        counts["negatives"] = len(files)
    print(f"Wrote {counts} slices to {args.out.resolve()}")


if __name__ == "__main__":
    main()
