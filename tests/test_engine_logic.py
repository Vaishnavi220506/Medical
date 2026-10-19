import numpy as np
import pytest

from oncovision.models.engine import Classification, Engine, Segmentation
from oncovision.models.registry import get_profile


def _seg(prob):
    labels = (prob > 0.5).astype(np.uint8)
    return Segmentation("model", labels, np.zeros_like(prob, dtype=np.float32), fg_prob=prob.astype(np.float32), fg_label=np.ones_like(labels))


def _cls(p_normal):
    probs = {"normal": p_normal, "no_tumor": p_normal, "benign": 1 - p_normal, "glioma": 1 - p_normal}
    return Classification("model", "x", "x", probs, 0.9, 0.1, 0.01)


def test_breast_threshold_is_not_lowered(engine):
    prob = np.zeros((1, 64, 64))
    prob[0, 20:40, 20:40] = 0.35
    prob[0, 28:32, 28:32] = 0.6
    seg = _seg(prob)
    assert engine.refine(seg, _cls(0.05), get_profile("breast_us")) is seg


@pytest.fixture()
def engine(tmp_path):
    return Engine(tmp_path)


def test_confident_normal_suppresses_outline(engine):
    prob = np.zeros((1, 64, 64))
    prob[0, 10:20, 10:20] = 0.95
    out = engine.refine(_seg(prob), _cls(0.9), get_profile("breast_us"))
    assert not out.labels.any() and "not reported" in out.note


def test_confident_tumor_lowers_threshold(engine):
    prob = np.zeros((1, 64, 64))
    prob[0, 20:40, 20:40] = 0.35  # below 0.5: missed by the default threshold
    prob[0, 28:32, 28:32] = 0.6
    out = engine.refine(_seg(prob), _cls(0.05), get_profile("brain_mri"))
    assert out.labels.sum() == 400 and out.threshold == 0.2 and out.note
    assert out.uncertainty[0, 25, 25] > 0.5  # weak pixels are marked uncertain
    assert set(np.unique(out.labels)) == {0, 1}


def test_uncertain_classifier_leaves_outline_alone(engine):
    prob = np.zeros((1, 64, 64))
    prob[0, 10:20, 10:20] = 0.9
    seg = _seg(prob)
    assert engine.refine(seg, _cls(0.5), get_profile("breast_us")) is seg


def test_faint_noise_is_not_promoted(engine):
    prob = np.full((1, 64, 64), 0.25)  # faint everywhere, no real peak
    out = engine.refine(_seg(prob), _cls(0.05), get_profile("brain_mri"))
    assert not out.labels.any()


def test_lowered_threshold_does_not_add_separate_blobs(engine):
    prob = np.zeros((1, 64, 64))
    prob[0, 5:15, 5:15] = 0.6  # the lesion the model found
    prob[0, 40:60, 40:60] = 0.3  # a separate faint region
    out = engine.refine(_seg(prob), _cls(0.05), get_profile("brain_mri"))
    assert out.labels[0, 50, 50] == 0 and out.labels[0, 10, 10] == 1


def test_diameters_use_largest_lesion():
    from oncovision.imaging.measure import measure

    labels = np.zeros((1, 200, 200), dtype=np.uint8)
    labels[0, 20:40, 20:40] = 1  # 20 x 20 lesion
    labels[0, 150:155, 150:155] = 1  # small separate spot far away
    ref = np.ones((1, 200, 200), dtype=np.float32)
    m = measure(labels, ref, get_profile("breast_us"), (1.0, 1.0, 1.0), axial_volume=False)
    assert m["lesion_count"] == 2 and m["longest_diameter"] < 30
