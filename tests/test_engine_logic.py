import numpy as np
import pytest

from oncovision.models.engine import Classification, Engine, Segmentation
from oncovision.models.registry import get_profile


def _seg(prob):
    labels = (prob > 0.5).astype(np.uint8)
    return Segmentation("model", labels, np.zeros_like(prob, dtype=np.float32), fg_prob=prob.astype(np.float32), fg_label=np.ones_like(labels))


def _cls(p_normal):
    return Classification("model", "x", "x", {"normal": p_normal, "benign": 1 - p_normal, "malignant": 0.0}, 0.9, 0.1, 0.01)


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
    out = engine.refine(_seg(prob), _cls(0.05), get_profile("breast_us"))
    assert out.labels.sum() == 400 and out.threshold == 0.2 and out.note
    assert out.uncertainty[0, 25, 25] > 0.5  # weak pixels are marked uncertain


def test_uncertain_classifier_leaves_outline_alone(engine):
    prob = np.zeros((1, 64, 64))
    prob[0, 10:20, 10:20] = 0.9
    seg = _seg(prob)
    assert engine.refine(seg, _cls(0.5), get_profile("breast_us")) is seg


def test_faint_noise_is_not_promoted(engine):
    prob = np.full((1, 64, 64), 0.25)  # faint everywhere, no real peak
    out = engine.refine(_seg(prob), _cls(0.05), get_profile("breast_us"))
    assert not out.labels.any()
