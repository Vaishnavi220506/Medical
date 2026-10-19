"""Combine three public breast ultrasound datasets into one training set.

* BUSI (Al-Dhabyani et al. 2020): 780 images, normal / benign / malignant, with masks.
* BUS-BRA (Gómez-Flores et al. 2024): 1,875 biopsy-proven benign / malignant images
  from 1,064 patients, with masks. https://zenodo.org/records/8231412 (CC-BY-4.0)
* BUS-UCLM (Vallez et al. 2025): 683 images from 38 patients, 419 of them normal,
  with masks. https://data.mendeley.com/datasets/7fvgj4jsp7 (CC-BY-4.0)

Writes, with patient-disjoint train and test sets::

    <out>/{train,test}/cls/{normal,benign,malignant}/<source>-<patient>__<image>.png
    <out>/{train,test}/seg/{images,masks}/<source>-<patient>__<image>.png

The BUSI test images are the validation images of the previously shipped BUSI-only
models (same seed and split), so old and new models can be compared on them.
BUSI has no patient IDs, so each BUSI image counts as its own patient.

Usage:
    python scripts/prepare_breast.py --download --busi data/busi --raw data/breast_raw --out data/breast
    python -m oncovision.training.train_classifier --profile breast_us --train-dir data/breast/train/cls \\
        --test-dir data/breast/test/cls --pretrained
    python -m oncovision.training.train_segmenter --profile breast_us --format maskfolder \\
        --data data/breast/train/seg/images --masks data/breast/train/seg/masks
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oncovision.training.datasets import find_busi_pairs, find_class_images, split_by_key

CLASSES = ["normal", "benign", "malignant"]
MAX_SIDE = 512
BUSBRA_URL = "https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content"
BUSUCLM_API = "https://data.mendeley.com/public-api/datasets/7fvgj4jsp7"


def download(raw: Path) -> None:
    bra = raw / "BUSBRA"
    if not (bra / "bus_data.csv").exists():
        raw.mkdir(parents=True, exist_ok=True)
        print("Downloading BUS-BRA (130 MB)...", flush=True)
        urllib.request.urlretrieve(BUSBRA_URL, raw / "BUSBRA.zip")
        with zipfile.ZipFile(raw / "BUSBRA.zip") as z:
            z.extractall(raw)
    uclm = raw / "BUS-UCLM"
    if not (uclm / "INFO.csv").exists():
        print("Downloading BUS-UCLM (330 MB, 1,367 files)...", flush=True)
        files = json.load(urllib.request.urlopen(BUSUCLM_API))["files"]
        # Images and masks share file names and sit in two folders; masks are the small files.
        sizes: dict[str, list[int]] = {}
        for f in files:
            sizes.setdefault(f.get("folder_id"), []).append(f["size"])
        kind = {k: ("masks" if sorted(v)[len(v) // 2] < 50_000 else "images") for k, v in sizes.items()}
        jobs = []
        for f in files:
            sub = "." if f["filename"].endswith(".csv") else kind[f.get("folder_id")]
            (uclm / sub).mkdir(parents=True, exist_ok=True)
            jobs.append((f["content_details"]["download_url"], uclm / sub / f["filename"]))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda job: urllib.request.urlretrieve(*job), jobs))


def _gray(path: Path) -> Image.Image:
    img = Image.open(path).convert("L")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    return img


def _mask(paths: list[Path], size: tuple[int, int]) -> Image.Image:
    m = np.zeros(size[::-1], dtype=bool)
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB").resize(size, Image.NEAREST))
        m |= arr.max(axis=2) > 127
    return Image.fromarray(m.astype(np.uint8) * 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--busi", type=Path, required=True, help="BUSI folder ({benign,malignant,normal}/...)")
    parser.add_argument("--raw", type=Path, default=Path("data/breast_raw"), help="Where BUS-BRA and BUS-UCLM are (or go)")
    parser.add_argument("--out", type=Path, default=Path("data/breast"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.download:
        download(args.raw)

    # (source, patient, name, class, image path, mask paths)
    records = []
    for path, masks in find_busi_pairs(args.busi):
        records.append(("busi", path.stem.replace(" ", ""), "0", path.parent.name.lower(), path, masks))
    bra = args.raw / "BUSBRA"
    for row in csv.DictReader(open(bra / "bus_data.csv")):
        name = row["ID"].removeprefix("bus_")
        records.append(("busbra", row["Case"], name, row["Pathology"], bra / "Images" / f"bus_{name}.png", [bra / "Masks" / f"mask_{name}.png"]))
    uclm = args.raw / "BUS-UCLM"
    for row in csv.DictReader(open(uclm / "INFO.csv", encoding="utf-8-sig"), delimiter=";"):
        stem = row["Image"].removesuffix(".png")
        records.append(("uclm", stem.split("_")[0], stem, row["Label"].lower(), uclm / "images" / row["Image"], [uclm / "masks" / row["Image"]]))

    # BUSI test = the validation images of the old BUSI-only classifier and segmenter.
    stem = lambda it: it[0].stem  # noqa: E731
    busi_val_cls = {p.stem.replace(" ", "") for p, _ in split_by_key(find_class_images(args.busi, CLASSES), stem, 0.15)[1]}
    busi_val_seg = {p.stem.replace(" ", "") for p, _ in split_by_key(find_busi_pairs(args.busi), stem, 0.15)[1]}
    rng = random.Random(42)
    test_patients = set()
    for source in ("busbra", "uclm"):
        patients = sorted({r[1] for r in records if r[0] == source})
        rng.shuffle(patients)
        test_patients |= {(source, p) for p in patients[: max(1, round(len(patients) * args.test_fraction))]}

    counts: dict[str, dict[str, int]] = {}
    for source, patient, name, cls, img_path, mask_paths in records:
        if source == "busi":
            split = "test" if patient in busi_val_cls | busi_val_seg else "train"
        else:
            split = "test" if (source, patient) in test_patients else "train"
        stem_out = f"{source}-{patient}__{name}"
        img = _gray(img_path)
        root = args.out / split
        for sub in (f"cls/{cls}", "seg/images", "seg/masks"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        img.save(root / "cls" / cls / f"{stem_out}.png")
        img.save(root / "seg" / "images" / f"{stem_out}.png")
        _mask(mask_paths, img.size).save(root / "seg" / "masks" / f"{stem_out}.png")
        key = f"{split}/{source}"
        counts.setdefault(key, {c: 0 for c in CLASSES})[cls] += 1

    meta = {
        "busi_old_classifier_val": sorted(f"busi-{s}__0" for s in busi_val_cls),
        "busi_old_segmenter_val": sorted(f"busi-{s}__0" for s in busi_val_seg),
        "counts": counts,
    }
    (args.out / "splits.json").write_text(json.dumps(meta, indent=1))
    for key, c in sorted(counts.items()):
        print(f"{key:14s} {c}")


if __name__ == "__main__":
    main()
