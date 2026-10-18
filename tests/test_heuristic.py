import numpy as np
from PIL import Image

from oncovision.imaging.io import load_scan
from oncovision.models import heuristic
from oncovision.models.engine import Engine
from oncovision.models.registry import get_profile


def test_brain_phantom_subregions(demo_dir):
    files = [(p.name, p.read_bytes(), None) for p in sorted(demo_dir.glob("brain_t0_*.nii.gz"))]
    scan = load_scan(files)
    assert set(scan.channels) == {"flair", "t1", "t1ce", "t2"}
    labels, unc, sal = heuristic.segment(scan.channels["t1ce"], get_profile("brain_mri"), flair=scan.channels["flair"])
    counts = {i: int((labels == i).sum()) for i in (1, 2, 3)}
    # Phantom truth: enhancing shell ~2550 voxels, necrosis ~510, edema ~12700.
    assert 1800 < counts[3] < 3300
    assert 300 < counts[1] < 800
    assert 8000 < counts[2] < 20000
    assert unc.max() <= 1 and sal.max() <= 1


def test_breast_ultrasound_lesion(demo_dir):
    img = np.asarray(Image.open(demo_dir / "breast_us.png").convert("L"), dtype=np.float32)[None]
    labels, _, _ = heuristic.segment(img, get_profile("breast_us"))
    yy, xx = np.mgrid[:360, :480]
    truth = ((yy - 170) / 45) ** 2 + ((xx - 250) / 65) ** 2 <= 1
    pred = labels[0] > 0
    assert 2 * (pred & truth).sum() / (pred.sum() + truth.sum()) > 0.8


def test_heuristic_router(demo_dir, tmp_path):
    engine = Engine(tmp_path)
    us = load_scan([("x.png", (demo_dir / "breast_us.png").read_bytes(), None)])
    brain = load_scan([("x.png", (demo_dir / "brain_slice.png").read_bytes(), None)])
    assert engine.route(us).profile == "breast_us"
    assert engine.route(brain).profile == "brain_mri"


def test_normal_scan_has_no_detection():
    yy, xx = np.mgrid[:128, :128]
    head = (((yy - 64) / 50) ** 2 + ((xx - 64) / 42) ** 2 <= 1).astype(np.float32) * 0.5
    rng = np.random.default_rng(0)
    img = (head + rng.normal(0, 0.01, head.shape) * head)[None]
    labels, _, _ = heuristic.segment(img, get_profile("brain_mri"))
    assert labels.sum() == 0
