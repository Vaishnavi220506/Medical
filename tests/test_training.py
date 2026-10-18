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
