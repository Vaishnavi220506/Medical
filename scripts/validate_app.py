"""End-to-end check of the whole app pipeline on held-out real scans.

Brain uses the dataset's own test set. Breast and lung use only the
validation split their classifiers held out during training.

Unlike the per-model evaluation (oncovision.training.evaluate), this runs every
image through exactly what the web app does: automatic scan-type routing,
segmentation, classification, measurements and review flags. It reports what
a user would actually experience.

Usage (any subset of the options):
    python scripts/validate_app.py \\
        --brain-test data/brain/Testing \\
        --busi data/busi \\
        --lung data/lidc_2d/cls \\
        --other data/router/other \\
        --per-class 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ONCOVISION_DATA_DIR", tempfile.mkdtemp(prefix="oncovision_validate_"))

from oncovision.analysis.pipeline import UnsupportedScanError, analyze, get_engine  # noqa: E402
from oncovision.models.registry import get_profile  # noqa: E402
from oncovision.training.datasets import IMAGE_EXT, canonical_class, find_class_images, split_by_key  # noqa: E402


def collect(root: Path, per_class: int, rng: random.Random, exclude_masks: bool = True) -> list[tuple[Path, str]]:
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT and not (exclude_masks and "_mask" in p.stem))
        rng.shuffle(files)
        items += [(f, canonical_class(folder.name)) for f in files[:per_class]]
    return items


def held_out(root: Path, profile_key: str, per_class: int, rng: random.Random, val_fraction: float = 0.15) -> list[tuple[Path, str]]:
    """Only images from the validation split the classifier never trained on (same split as train_classifier)."""
    classes = list(get_profile(profile_key).classes)
    items = find_class_images(root, classes)
    _, val = split_by_key(items, lambda it: it[0].stem.split("__")[0], val_fraction)
    by_class: dict[int, list] = defaultdict(list)
    for path, label in val:
        by_class[label].append(path)
    out = []
    for label, paths in sorted(by_class.items()):
        rng.shuffle(paths)
        out += [(p, classes[label]) for p in paths[:per_class]]
    return out


def run(items: list[tuple[Path, str]], expected_type: str | None, normal_class: str | None) -> dict:
    stats: dict = defaultdict(int)
    confusion: Counter = Counter()
    seconds = []
    for path, truth in items:
        stats["n"] += 1
        start = time.time()
        try:
            study = analyze([(path.name, path.read_bytes(), None)], "auto", "validation")
        except UnsupportedScanError:
            stats["rejected"] += 1
            confusion[(truth, "rejected")] += 1
            continue
        finally:
            seconds.append(time.time() - start)
        routed = study["profile"]["key"]
        stats["routed_correctly"] += routed == expected_type
        if routed != expected_type:
            confusion[(truth, f"routed:{routed}")] += 1
            continue
        pred = study["classification"]["label"]
        confusion[(truth, pred)] += 1
        stats["class_correct"] += pred == truth
        detected = study["measurements"].get("detected", False)
        is_normal = truth == normal_class
        stats["normal_n" if is_normal else "tumor_n"] += 1
        if is_normal:
            stats["normal_seg_false_positive"] += detected
        else:
            stats["tumor_seg_detected"] += detected
        stats["flagged_for_review"] += study["review"]["required"]
    n = max(stats["n"], 1)
    routed_n = max(stats["routed_correctly"], 1)
    return {
        "images": stats["n"],
        "rejected_as_unsupported": stats["rejected"],
        "routing_accuracy": round(stats["routed_correctly"] / n, 3) if expected_type else None,
        "classification_accuracy_when_routed": round(stats["class_correct"] / routed_n, 3) if expected_type else None,
        "tumor_outlined_rate": round(stats["tumor_seg_detected"] / max(stats["tumor_n"], 1), 3) if stats["tumor_n"] else None,
        "normal_false_outline_rate": round(stats["normal_seg_false_positive"] / max(stats["normal_n"], 1), 3) if stats["normal_n"] else None,
        "flagged_for_review_rate": round(stats["flagged_for_review"] / routed_n, 3) if expected_type else None,
        "seconds_per_scan": round(sum(seconds) / max(len(seconds), 1), 2),
        "confusion": {f"{t} -> {p}": c for (t, p), c in sorted(confusion.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain-test", type=Path)
    parser.add_argument("--busi", type=Path)
    parser.add_argument("--lung", type=Path)
    parser.add_argument("--other", type=Path, help="Folder of images that should be rejected")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("reports/app_validation.json"))
    args = parser.parse_args()

    rng = random.Random(0)
    print("Models:", json.dumps({k: {m: v[m] for m in ("classifier", "segmenter")} for k, v in get_engine().status()["profiles"].items()}),
          "router:", get_engine().status()["router"])
    results = {}
    if args.brain_test:
        results["brain_mri"] = run(collect(args.brain_test, args.per_class, rng), "brain_mri", "no_tumor")
    if args.busi:
        results["breast_us"] = run(held_out(args.busi, "breast_us", args.per_class, rng), "breast_us", "normal")
    if args.lung:
        results["lung_ct"] = run(held_out(args.lung, "lung_ct", args.per_class, rng), "lung_ct", "normal")
    if args.other:
        files = sorted(p for p in args.other.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        rng.shuffle(files)
        results["other"] = run([(f, "other") for f in files[: args.per_class * 2]], None, None)
    for name, r in results.items():
        print(f"\n== {name}")
        print(json.dumps(r, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
