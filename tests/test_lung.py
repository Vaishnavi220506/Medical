"""Lung CT: lung finding, nodule measurements, Lung-RADS and the 3D model path (tiny untrained model)."""

import numpy as np
import pytest
import torch

from oncovision.imaging.io import Scan
from oncovision.imaging.lung import lung_box, lung_mask
from oncovision.imaging.measure import lung_rads, measure
from oncovision.models.engine import Engine
from oncovision.models.networks import UNet3D
from oncovision.models.registry import get_profile


def chest(z=40, n=96):
    """Synthetic chest in the lung window scaled to [0, 1]: air 0, body 0.7, two lungs 0.1, a nodule."""
    zz, yy, xx = np.mgrid[:z, :n, :n]
    vol = np.zeros((z, n, n), np.float32)
    vol[((yy - 48) / 40) ** 2 + ((xx - 48) / 44) ** 2 <= 1] = 0.7
    for cx in (28, 68):
        vol[(((yy - 46) / 26) ** 2 + ((xx - cx) / 14) ** 2 <= 1) & (zz > 4) & (zz < 36)] = 0.1
    nodule = (zz - 20) ** 2 + (yy - 46) ** 2 + (xx - 68) ** 2 <= 4**2
    vol[nodule] = 0.7
    return vol, nodule


def test_lung_mask_finds_lungs_and_fills_nodules():
    vol, nodule = chest()
    lungs = lung_mask(vol)
    assert lungs[20, 46, 28] and lungs[20, 46, 68]  # both lungs
    assert lungs[nodule].all()  # nodule inside the lung is filled in
    assert not lungs[20, 2, 2] and not lungs[20, 46, 48]  # outside air, mediastinum
    box = lung_box(vol, margin=2)
    assert box[1].start <= 20 and box[2].start <= 14 and box[2].stop >= 82


def test_lung_rads_categories():
    assert [lung_rads(d)["category"] for d in (4, 6, 7.9, 8, 14.9, 15, 30)] == ["2", "3", "3", "4A", "4A", "4B", "4B"]


def test_nodule_measurements():
    vol, nodule = chest()
    labels = np.zeros(vol.shape, np.uint8)
    labels[nodule] = 1
    small = np.zeros_like(nodule)
    small[30, 40:42, 26:28] = True
    labels[small] = 1
    m = measure(labels, vol, get_profile("lung_ct"), (1.0, 1.0, 1.0), axial_volume=True)
    big, little = m["nodules"]
    assert big["side"] == "left lung" and little["side"] == "right lung"  # image right = patient left
    assert 8 <= big["longest_diameter"] <= 10 and big["volume_mm3"] == pytest.approx(nodule.sum())
    assert m["lung_rads"]["category"] == "4A"


def test_3d_lung_path_filters_findings(tmp_path):
    """Plumbing of the 3D lung path: findings outside the lungs or under 3 mm are dropped."""
    model = UNet3D(1, 2, base=4)
    meta = {"architecture": "unet3d", "channels": ["image"], "num_classes": 2, "label_indices": [0, 1], "base": 4,
            "patch": 32, "spacing_mm": 1.0, "crop": "lungs", "threshold": 0.5, "min_component_voxels": 14}
    (tmp_path / "lung_ct").mkdir()
    torch.save({"state_dict": model.state_dict(), "meta": meta}, tmp_path / "lung_ct" / "segmenter3d.pt")
    engine = Engine(tmp_path, device="cpu")
    profile = get_profile("lung_ct")
    vol, nodule = chest()
    scan = Scan(channels={"image": vol * 1400 - 1000}, spacing=(1.0, 1.0, 1.0), source_format="nifti")
    assert engine.needs_volume("lung_ct") and engine.has_models("lung_ct")

    # Replace the network's output: a real nodule, a speck under 3 mm, and a blob in the mediastinum.
    fake = np.zeros(vol.shape, np.float32)
    fake[nodule] = 1
    fake[30, 46, 28] = 1
    fake[18:23, 44:49, 46:51] = 1

    import oncovision.models.engine as eng

    box = eng.lung_box(eng.channel_stack(scan, ("image",))[0])  # the crop the engine will use
    original = eng.sliding_window_3d
    eng.sliding_window_3d = lambda m, t, **kw: torch.from_numpy(np.stack([1 - fake, fake])[(slice(None), *box)])
    try:
        seg = engine.segment(scan, profile)
    finally:
        eng.sliding_window_3d = original
    assert seg.model == "3d"
    assert seg.labels[nodule].all()
    assert seg.labels.sum() == nodule.sum()  # speck and mediastinal blob removed
    cls, _ = engine.classify_from_3d(seg, profile)
    assert cls.label == "tumor" and "nodule" in cls.display.lower()
