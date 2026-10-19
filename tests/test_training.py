"""Smoke tests: each training script runs on a tiny dataset and its checkpoint is used by the app."""

import sys

import nibabel as nib
import numpy as np
from PIL import Image

from oncovision.analysis.pipeline import analyze, get_engine
from oncovision.training import train_classifier, train_segmenter


def _run(module, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    module.main()


def test_classifier_training(isolated, demo_dir, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    root = tmp_path / "Training"
    for folder in ("glioma", "meningioma", "notumor", "pituitary"):
        (root / folder).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(root / folder / f"{i}.jpg")
    out = isolated.weights_dir / "brain_mri" / "classifier.pt"
    _run(train_classifier, ["--profile", "brain_mri", "--train-dir", str(root), "--test-dir", str(root), "--epochs", "1", "--batch-size", "4", "--workers", "0", "--out", str(out)], monkeypatch)
    assert out.exists()

    study = analyze([("s.png", (demo_dir / "brain_slice.png").read_bytes(), None)], "brain_mri", "Case-T")
    assert study["classification"]["method"] == "model"
    assert set(study["classification"]["probabilities"]) == {"no_tumor", "glioma", "meningioma", "pituitary"}
    assert study["mode"] == "mixed"
    assert get_engine().status()["profiles"]["brain_mri"]["classifier_metrics"]["test"]["n"] == 16


def test_busi_segmenter_training(isolated, demo_dir, tmp_path, monkeypatch):
    root = tmp_path / "BUSI" / "benign"
    root.mkdir(parents=True)
    img = Image.open(demo_dir / "breast_us.png")
    yy, xx = np.mgrid[:360, :480]
    mask = ((((yy - 170) / 45) ** 2 + ((xx - 250) / 65) ** 2) <= 1).astype(np.uint8) * 255
    for i in range(3):
        img.save(root / f"benign ({i}).png")
        Image.fromarray(mask).save(root / f"benign ({i})_mask.png")
    out = isolated.weights_dir / "breast_us" / "segmenter.pt"
    _run(train_segmenter, ["--profile", "breast_us", "--format", "busi", "--data", str(root.parent), "--epochs", "1", "--batch-size", "2", "--base", "8", "--workers", "0", "--val-fraction", "0.34", "--out", str(out)], monkeypatch)
    assert out.exists()
    study = analyze([("us.png", (demo_dir / "breast_us.png").read_bytes(), None)], "breast_us", "Case-U")
    assert study["segmentation"]["method"] == "model"


def test_brats_segmenter_training(isolated, demo, tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    root = tmp_path / "brats"
    for c in range(2):
        case = root / f"BraTS2021_0000{c}"
        case.mkdir(parents=True)
        vols = demo.phantom(9.0 + c, rng)
        for seq, vol in vols.items():
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(case / f"{case.name}_{seq}.nii.gz"))
        seg = np.zeros(vol.shape, dtype=np.uint8)
        seg[vols["flair"] > 750] = 2
        seg[vols["t1ce"] > 850] = 4  # BraTS 2021 label for enhancing tumor
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(case / f"{case.name}_seg.nii.gz"))
    out = isolated.weights_dir / "brain_mri" / "segmenter.pt"
    _run(train_segmenter, ["--profile", "brain_mri", "--format", "brats", "--data", str(root), "--prepared-dir", str(tmp_path / "slices"), "--epochs", "1", "--batch-size", "8", "--base", "8", "--workers", "0", "--val-fraction", "0.5", "--out", str(out)], monkeypatch)
    labels = {int(v) for f in (tmp_path / "slices").glob("*.npz") for v in np.unique(np.load(f)["label"])}
    assert labels <= {0, 1, 2, 3} and 3 in labels

    files = [(f"x_{s}.nii.gz", nib.Nifti1Image(v, np.eye(4)).to_bytes(), s) for s, v in demo.phantom(10.0, rng).items()]
    study = analyze(files, "brain_mri", "Case-B")
    assert study["segmentation"]["method"] == "model"


def test_binary_brain_segmenter_and_evaluate(isolated, demo_dir, tmp_path, monkeypatch):
    """Figshare-style binary masks map to the whole-tumor label, and evaluate.py writes report files."""
    from oncovision.training import evaluate

    img = np.asarray(Image.open(demo_dir / "brain_slice.png"))
    mask = (img > 200).astype(np.uint8) * 255
    for d in ("images", "masks"):
        (tmp_path / "fig" / d).mkdir(parents=True)
    for pid in range(3):
        Image.fromarray(img).save(tmp_path / "fig" / "images" / f"{pid}__1.png")
        Image.fromarray(mask).save(tmp_path / "fig" / "masks" / f"{pid}__1.png")
    out = isolated.weights_dir / "brain_mri" / "segmenter.pt"
    _run(train_segmenter, ["--profile", "brain_mri", "--format", "maskfolder", "--data", str(tmp_path / "fig" / "images"), "--masks", str(tmp_path / "fig" / "masks"), "--epochs", "1", "--batch-size", "2", "--base", "8", "--workers", "0", "--val-fraction", "0.34", "--out", str(out)], monkeypatch)
    import torch

    meta = torch.load(out, weights_only=True)["meta"]
    assert meta["label_indices"] == [0, 4] and meta["channels"] == ["image"]

    study = analyze([("s.png", (demo_dir / "brain_slice.png").read_bytes(), None)], "brain_mri", "Case-F")
    assert study["segmentation"]["method"] == "model"
    assert set(np.unique(get_store_labels(study))) <= {0, 4}

    report_dir = tmp_path / "report"
    _run(evaluate, ["segmenter", "--profile", "brain_mri", "--format", "maskfolder", "--data", str(tmp_path / "fig" / "images"), "--masks", str(tmp_path / "fig" / "masks"), "--weights", str(out), "--out", str(report_dir), "--workers", "0"], monkeypatch)
    assert (report_dir / "metrics.json").exists() and (report_dir / "segmentation_examples.png").exists()


def test_classifier_evaluate_report(isolated, tmp_path, monkeypatch):
    from oncovision.training import evaluate

    rng = np.random.default_rng(0)
    root = tmp_path / "Testing"
    for folder in ("normal", "benign", "malignant"):
        (root / folder).mkdir(parents=True)
        for i in range(3):
            Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(root / folder / f"{i}.png")
    out = tmp_path / "clf.pt"
    _run(train_classifier, ["--profile", "breast_us", "--train-dir", str(root), "--epochs", "1", "--batch-size", "4", "--workers", "0", "--out", str(out)], monkeypatch)
    report_dir = tmp_path / "report"
    _run(evaluate, ["classifier", "--profile", "breast_us", "--data", str(root), "--weights", str(out), "--out", str(report_dir), "--workers", "0"], monkeypatch)
    for name in ("metrics.json", "confusion_matrix.png", "gradcam_examples.png"):
        assert (report_dir / name).exists()


def get_store_labels(study):
    from oncovision.analysis.pipeline import get_store

    return get_store().arrays(study["id"])["labels"]


def test_untrained_scan_type_is_rejected_unless_chosen(isolated, demo_dir, tmp_path, monkeypatch):
    """Once real models exist, auto-detected scan types without a model are refused instead of guessed."""
    import pytest

    from oncovision.analysis.pipeline import UnsupportedScanError

    rng = np.random.default_rng(0)
    root = tmp_path / "Training"
    for folder in ("glioma", "meningioma", "notumor", "pituitary"):
        (root / folder).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(root / folder / f"{i}.jpg")
    _run(train_classifier, ["--profile", "brain_mri", "--train-dir", str(root), "--epochs", "1", "--batch-size", "4", "--workers", "0", "--val-fraction", "0.25", "--out", str(isolated.weights_dir / "brain_mri" / "classifier.pt")], monkeypatch)
    us = [("us.png", (demo_dir / "breast_us.png").read_bytes(), None)]
    with pytest.raises(UnsupportedScanError, match="no trained model"):
        analyze(us, "auto", "Case-R")
    assert analyze(us, "breast_us", "Case-R")["mode"] == "demo"


def test_3d_segmenter_training_and_inference(isolated, demo, tmp_path, monkeypatch):
    """The 3D U-Net trains on BraTS-2023-style folders and is used for multi-sequence volumes."""
    from oncovision.training import train_segmenter3d

    rng = np.random.default_rng(3)
    root = tmp_path / "brats23"
    names = {"flair": "t2f", "t1": "t1n", "t1ce": "t1c", "t2": "t2w"}
    for c in range(3):
        case = root / f"BraTS-GLI-0000{c}-000"
        case.mkdir(parents=True)
        vols = demo.phantom(9.0 + c, rng)
        for seq, vol in vols.items():
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(case / f"{case.name}-{names[seq]}.nii.gz"))
        seg = np.zeros(vol.shape, dtype=np.uint8)
        seg[vols["flair"] > 750] = 2
        seg[vols["t1ce"] > 850] = 3
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(case / f"{case.name}-seg.nii.gz"))
    out = isolated.weights_dir / "brain_mri" / "segmenter3d.pt"
    _run(train_segmenter3d, ["--data", str(root), "--prepared-dir", str(tmp_path / "prep3d"), "--epochs", "1", "--samples-per-epoch", "2", "--patch", "32", "--batch-size", "1", "--base", "4", "--val-cases", "1", "--workers", "0", "--out", str(out)], monkeypatch)
    import torch

    meta = torch.load(out, weights_only=True)["meta"]
    assert meta["architecture"] == "unet3d" and "whole_tumor" in meta["metrics"]["validation"]

    files = [(f"x_{s}.nii.gz", nib.Nifti1Image(v, np.eye(4)).to_bytes(), s) for s, v in demo.phantom(10.0, rng).items()]
    study = analyze(files, "brain_mri", "Case-3D")
    assert study["segmentation"]["model"] == "3d"
    # For 3D volumes the 3D model decides tumor presence (the 2D classifier is not used).
    assert study["classification"]["label"] in ("tumor", "no_tumor")
    single = analyze([("x_t1ce.nii.gz", files[2][1], "t1ce")], "brain_mri", "Case-3D")
    assert single["segmentation"]["model"] == "3d"  # one named sequence is enough
    unlabeled = analyze([("scan.nii.gz", files[2][1], None)], "brain_mri", "Case-3D")
    assert unlabeled["segmentation"]["model"] != "3d"  # unknown sequence -> 2D model
