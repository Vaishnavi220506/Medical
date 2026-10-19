"""Loads trained weights and runs routing, segmentation and classification.

Weights live under ``weights/``::

    weights/router.pt
    weights/<profile>/classifier.pt
    weights/<profile>/segmenter.pt

Each checkpoint is ``{"state_dict": ..., "meta": {...}}`` as written by the
training scripts. Any missing model falls back to the heuristic demo mode for
that step only, so you can e.g. train just the brain classifier first.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from ..imaging.io import Scan
from ..imaging.preprocess import foreground_mask, normalize, resize2d
from . import heuristic
from .networks import Classifier, UNet, UNet3D, enable_mc_dropout, grad_cam, sliding_window_3d
from .registry import PROFILES, ROUTER_CLASSES, CancerProfile

log = logging.getLogger(__name__)

MAX_CAM_SLICES = 48
BATCH = 16
GATE_NORMAL = 0.8  # no outline is reported when the classifier gives "normal" a higher probability
MIN_TUMOR_PIXELS = 20


@dataclass
class RouteResult:
    profile: str | None
    confidence: float
    method: str
    probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Classification:
    method: str
    label: str
    display: str
    probabilities: dict[str, float]
    confidence: float
    uncertainty: float  # normalized predictive entropy, 0 (certain) .. 1 (uniform)
    mc_std: float | None = None
    slices_used: list[int] = field(default_factory=list)


@dataclass
class Segmentation:
    method: str
    labels: np.ndarray  # (Z, H, W) uint8 label map
    uncertainty: np.ndarray  # (Z, H, W) float32 in [0, 1]
    saliency: np.ndarray | None = None  # heuristic contrast map, reused as demo heatmap
    fg_prob: np.ndarray | None = None  # (Z, H, W) probability of "any tumor" (trained models only)
    fg_label: np.ndarray | None = None  # (Z, H, W) most likely tumor label if the pixel were tumor
    threshold: float = 0.5
    note: str = ""
    model: str = ""  # which trained segmenter produced it (single-image / multi-sequence / 3d)


def _pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _entropy(p: np.ndarray, axis: int) -> np.ndarray:
    k = p.shape[axis]
    return -(p * np.log(np.clip(p, 1e-8, 1))).sum(axis=axis) / math.log(k)


def channel_stack(scan: Scan, names: tuple[str, ...]) -> np.ndarray:
    """Normalized (C, Z, H, W) input; missing sequences are filled with the primary one."""
    fallback = normalize(scan.primary())
    if names == ("image",):
        return fallback[None]
    return np.stack([normalize(scan.channels[n]) if n in scan.channels else fallback for n in names])


class Engine:
    def __init__(self, weights_dir: Path, device: str = "auto", mc_samples: int = 8):
        self.weights_dir = Path(weights_dir)
        self.device = _pick_device(device)
        self.mc_samples = max(1, mc_samples)
        self.router: tuple[Classifier, dict] | None = None
        self.classifiers: dict[str, tuple[Classifier, dict]] = {}
        self.segmenters: dict[str, tuple[UNet, dict]] = {}
        self.multiseq_segmenters: dict[str, tuple[UNet, dict]] = {}
        self.segmenters3d: dict[str, tuple[UNet3D, dict]] = {}
        self.reload()

    # ------------------------------------------------------------------ loading
    def _load(self, path: Path, build):
        if not path.is_file():
            return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            meta = ckpt.get("meta", {})
            model = build(meta)
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            log.info("Loaded %s", path)
            return model, meta
        except Exception:
            log.exception("Failed to load %s; using demo mode for it", path)
            return None

    def reload(self) -> None:
        self.router = self._load(
            self.weights_dir / "router.pt",
            lambda m: Classifier(num_classes=len(m.get("classes", ROUTER_CLASSES))),
        )
        self.classifiers, self.segmenters, self.multiseq_segmenters, self.segmenters3d = {}, {}, {}, {}
        for key, profile in PROFILES.items():
            clf = self._load(
                self.weights_dir / key / "classifier.pt",
                lambda m, p=profile: Classifier(num_classes=len(m.get("classes", p.classes))),
            )
            if clf:
                self.classifiers[key] = clf
            seg = self._load(
                self.weights_dir / key / "segmenter.pt",
                lambda m, p=profile: UNet(
                    in_channels=len(m.get("channels", p.seg_channels)),
                    num_classes=m.get("num_classes", p.num_seg_classes),
                    base=m.get("base", 32),
                ),
            )
            if seg:
                self.segmenters[key] = seg
            # Optional second segmenter for multi-sequence MRI (BraTS: FLAIR, T1, T1ce, T2),
            # which separates tumor sub-regions. Used when the upload has 2+ of its sequences.
            multi = self._load(
                self.weights_dir / key / "segmenter_multiseq.pt",
                lambda m, p=profile: UNet(
                    in_channels=len(m.get("channels", p.seg_channels)),
                    num_classes=m.get("num_classes", p.num_seg_classes),
                    base=m.get("base", 32),
                ),
            )
            if multi:
                self.multiseq_segmenters[key] = multi
            # Optional true 3D model (3D U-Net on BraTS) for multi-sequence volumes.
            seg3d = self._load(
                self.weights_dir / key / "segmenter3d.pt",
                lambda m, p=profile: UNet3D(len(m["channels"]), m["num_classes"], base=m.get("base", 16)),
            )
            if seg3d:
                self.segmenters3d[key] = seg3d

    def _segmenter_for(self, scan: Scan, profile: CancerProfile):
        """Most specific model first: 3D for multi-sequence volumes, then multi-sequence 2D, then single-image."""
        seg3d = self.segmenters3d.get(profile.key)
        if seg3d and scan.is_volume and sum(c in scan.channels for c in seg3d[1]["channels"]) >= 2:
            return seg3d, "3d"
        multi = self.multiseq_segmenters.get(profile.key)
        if multi and sum(c in scan.channels for c in multi[1].get("channels", ())) >= 2:
            return multi, "multi-sequence"
        if profile.key in self.segmenters:
            return self.segmenters[profile.key], "single-image"
        return (multi, "multi-sequence") if multi else (None, "")

    def status(self) -> dict:
        out = {"device": str(self.device), "router": self.router is not None, "profiles": {}}
        for key in PROFILES:
            clf, seg = self.classifiers.get(key), self.segmenters.get(key)
            out["profiles"][key] = {
                "classifier": clf is not None,
                "segmenter": seg is not None,
                "classifier_metrics": clf[1].get("metrics", {}) if clf else {},
                "segmenter_metrics": seg[1].get("metrics", {}) if seg else {},
                "segmenter3d": key in self.segmenters3d,
                "segmenter3d_metrics": self.segmenters3d[key][1].get("metrics", {}) if key in self.segmenters3d else {},
                "multiseq_segmenter": key in self.multiseq_segmenters,
                "multiseq_segmenter_metrics": self.multiseq_segmenters[key][1].get("metrics", {})
                if key in self.multiseq_segmenters
                else {},
            }
        return out

    def has_models(self, profile_key: str) -> bool:
        return any(profile_key in d for d in (self.classifiers, self.segmenters, self.multiseq_segmenters, self.segmenters3d))

    @property
    def demo_mode(self) -> bool:
        return not (self.classifiers or self.segmenters or self.multiseq_segmenters or self.segmenters3d)

    # ------------------------------------------------------------------ routing
    def route(self, scan: Scan) -> RouteResult:
        if self.router is not None:
            model, meta = self.router
            classes = meta.get("classes", list(ROUTER_CLASSES))
            size = meta.get("image_size", 224)
            vol = normalize(scan.primary())
            idx = int(np.argmax([foreground_mask(s).sum() for s in vol])) if vol.shape[0] > 1 else 0
            x = torch.from_numpy(resize2d(vol[idx], (size, size))[None, None].copy()).float().to(self.device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            best = int(np.argmax(probs))
            label = classes[best]
            return RouteResult(
                profile=None if label == "other" else label,
                confidence=float(probs[best]),
                method="model",
                probabilities={c: float(p) for c, p in zip(classes, probs)},
                reason="Scan type predicted by the router model.",
            )
        return self._heuristic_route(scan)

    def _heuristic_route(self, scan: Scan) -> RouteResult:
        meta = {k: v.upper() for k, v in scan.metadata.items()}
        modality, body = meta.get("Modality", ""), meta.get("BodyPartExamined", "")
        if modality == "MR" or {"flair", "t1ce", "t2"} & set(scan.channels):
            return RouteResult("brain_mri", 0.9, "metadata", reason="MRI sequences / DICOM modality MR.")
        if modality == "CT" or "CHEST" in body or "LUNG" in body:
            return RouteResult("lung_ct", 0.85, "metadata", reason="DICOM modality CT / chest.")
        if modality == "US" or "BREAST" in body:
            return RouteResult("breast_us", 0.85, "metadata", reason="DICOM modality ultrasound / breast.")
        if scan.source_format == "nifti":
            return RouteResult("brain_mri", 0.6, "heuristic", reason="NIfTI volumes are most often brain MRI.")

        # Plain image: look at the shape of the imaged body part.
        vol = normalize(scan.primary())
        sl = vol[vol.shape[0] // 2]
        fg = foreground_mask(sl)
        if fg.sum() < 0.05 * fg.size:
            return RouteResult("breast_us", 0.4, "heuristic", reason="Small field of view, guessed ultrasound.")
        inner = ndimage.binary_erosion(fg, iterations=5)
        mean = ndimage.uniform_filter(sl, 5)
        std = np.sqrt(np.clip(ndimage.uniform_filter(sl**2, 5) - mean**2, 0, None))
        speckle = float(np.median(std[inner] / (mean[inner] + 0.05))) if inner.any() else 0.0
        if speckle > 0.08:
            return RouteResult("breast_us", 0.55, "heuristic", reason="Grainy speckle texture typical of ultrasound.")
        dark_inside = float(((sl < 0.25) & ndimage.binary_erosion(fg, iterations=5)).sum() / max(fg.sum(), 1))
        hull_fill = fg.sum() / max(ndimage.binary_fill_holes(ndimage.binary_closing(fg, iterations=15)).sum(), 1)
        if dark_inside > 0.2:
            return RouteResult("lung_ct", 0.55, "heuristic", reason="Large dark regions inside the body (lung fields).")
        if hull_fill > 0.9:
            return RouteResult("brain_mri", 0.55, "heuristic", reason="Compact, rounded body outline (head).")
        return RouteResult("breast_us", 0.45, "heuristic", reason="Irregular field of view, guessed ultrasound.")

    # ------------------------------------------------------------------ segmentation
    def segment(self, scan: Scan, profile: CancerProfile) -> Segmentation:
        chosen, model_name = self._segmenter_for(scan, profile)
        if chosen is None:
            labels, uncertainty, saliency = heuristic.segment(
                scan.primary(profile.classifier_channel_preference), profile, flair=scan.channels.get("flair")
            )
            return Segmentation("heuristic", labels, uncertainty, saliency)

        if model_name == "3d":
            return self._segment3d(scan, profile, *chosen)
        model, meta = chosen
        channels = tuple(meta.get("channels", profile.seg_channels))
        size = meta.get("image_size", profile.seg_size)
        # Model output class i -> label value (binary whole-tumor models map 1 -> the "tumor" label).
        to_label = np.array(meta.get("label_indices", range(profile.num_seg_classes)), dtype=np.uint8)
        x = channel_stack(scan, channels)  # (C, Z, H, W)
        _, z, h, w = x.shape
        labels = np.zeros((z, h, w), dtype=np.uint8)
        uncertainty = np.zeros((z, h, w), dtype=np.float32)
        fg_prob = np.zeros((z, h, w), dtype=np.float32)
        fg_label = np.zeros((z, h, w), dtype=np.uint8)
        active = [i for i in range(z) if x[:, i].max() > 0.05]

        def store(idx, probs):
            labels[idx] = to_label[probs.argmax(1)]
            fg_prob[idx] = 1.0 - probs[:, 0]
            fg_label[idx] = to_label[1 + probs[:, 1:].argmax(1)]

        def run(indices: list[int], samples: int):
            batch = np.stack([resize2d(x[:, i], (size, size)) for i in indices])
            t = torch.from_numpy(batch).float().to(self.device)
            with torch.no_grad():
                if samples == 1:
                    model.eval()
                    probs = torch.softmax(model(t), dim=1)[None]
                else:
                    enable_mc_dropout(model)
                    probs = torch.stack([torch.softmax(model(t), dim=1) for _ in range(samples)])
                    model.eval()
                probs = F.interpolate(probs.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False)
            return probs.view(samples, len(indices), -1, h, w).mean(0).cpu().numpy()

        # Fast deterministic pass everywhere, then MC dropout where tumor was found.
        for start in range(0, len(active), BATCH):
            idx = active[start : start + BATCH]
            store(idx, run(idx, 1))
        # Slices that contain tumor even at a low threshold get the MC-dropout pass.
        candidates = np.where((fg_prob > (profile.guided_low_threshold or 0.5)).any(axis=(1, 2)))[0]
        tumor_slices = sorted({j for i in candidates for j in (i - 1, i, i + 1) if 0 <= j < z})
        for start in range(0, len(tumor_slices), BATCH):
            idx = tumor_slices[start : start + BATCH]
            probs = run(idx, self.mc_samples)
            store(idx, probs)
            uncertainty[idx] = _entropy(probs, axis=1)
        return Segmentation("model", labels, uncertainty, fg_prob=fg_prob, fg_label=fg_label, model=model_name)

    def _segment3d(self, scan: Scan, profile: CancerProfile, model, meta) -> Segmentation:
        """Whole-volume 3D U-Net with sliding windows, on a 1 mm grid like BraTS."""
        x = channel_stack(scan, tuple(meta["channels"]))  # (C, Z, H, W)
        shape = x.shape[1:]
        target = meta.get("spacing_mm", 1.0)
        factors = [s / target for s in scan.spacing] if scan.spacing else [1.0, 1.0, 1.0]
        if any(abs(f - 1) > 0.1 for f in factors):
            x = np.stack([ndimage.zoom(c, factors, order=1) for c in x])
        # Crop to the brain exactly as in training (train_segmenter3d.prepare): the model's
        # instance normalization was calibrated on brain-cropped volumes, and running it on a
        # mostly-empty full volume degrades it badly (Dice 0.96 -> 0.44 on a test patient).
        brain = x.max(0) > 0.02
        probs_full = np.zeros((meta["num_classes"], *x.shape[1:]), dtype=np.float32)
        probs_full[0] = 1.0
        if brain.any():
            idx = [np.where(brain.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
            crop = tuple(slice(max(i[0] - 4, 0), i[-1] + 5) for i in idx)
            t = torch.from_numpy(np.ascontiguousarray(x[(slice(None), *crop)])).float().to(self.device)
            p = sliding_window_3d(model, t, patch=meta.get("patch", 128), overlap=0.25, flip_tta=self.device.type != "cpu")
            probs_full[(slice(None), *crop)] = p.cpu().numpy()
        probs = probs_full
        if probs.shape[1:] != shape:
            probs = np.stack([ndimage.zoom(p, [a / b for a, b in zip(shape, p.shape)], order=1) for p in probs])[:, : shape[0], : shape[1], : shape[2]]
        probs = probs / np.clip(probs.sum(0, keepdims=True), 1e-6, None)
        to_label = np.array(meta.get("label_indices", range(probs.shape[0])), dtype=np.uint8)
        labels = to_label[probs.argmax(0)]
        fg_prob = (1.0 - probs[0]).astype(np.float32)
        fg_label = to_label[1 + probs[1:].argmax(0)]
        # Drop isolated speckles: components under 10% of the largest tumor component.
        comp, n = ndimage.label(labels > 0)
        if n > 1:
            sizes = ndimage.sum(labels > 0, comp, range(1, n + 1))
            labels[np.isin(comp, np.where(sizes < 0.1 * sizes.max())[0] + 1)] = 0
        uncertainty = _entropy(probs, axis=0).astype(np.float32)
        uncertainty[(labels == 0) & (fg_prob <= 0.2)] = 0
        return Segmentation("model", labels.astype(np.uint8), uncertainty, fg_prob=fg_prob, fg_label=fg_label, model="3d")

    def refine(self, seg: Segmentation, cls: Classification, profile: CancerProfile) -> Segmentation:
        """Use the classifier, the more reliable model, to guide the tumor outline.

        - Classifier confident a tumor is present: lower the outline threshold
          (hysteresis) so weak but real tumor signal is kept, for scan types where
          that was shown to help (CancerProfile.guided_low_threshold).
        - Classifier confident the scan is normal: do not report an outline;
          the region is kept out of the measurements and explained in a note.

        Thresholds were chosen on held-out brain MRIs and breast ultrasounds,
        where gating cut false outlines on normal scans from 50% to 20%
        (breast) and 6% to 2% (brain) while losing at most 1 of 230 tumors.
        """
        if seg.fg_prob is None or cls.method != "model" or profile.normal_class not in cls.probabilities:
            return seg
        p_normal = cls.probabilities[profile.normal_class]
        if p_normal > GATE_NORMAL:
            if not seg.labels.any():
                return seg
            note = (
                f"The outline model marked a small region, but the classifier is {p_normal:.0%} confident this scan is "
                "normal, so it is not reported as a tumor. Look at the AI heatmap if you want to check it."
            )
            return Segmentation(seg.method, np.zeros_like(seg.labels), seg.uncertainty, seg.saliency, seg.fg_prob, seg.fg_label, 1.0, note, seg.model)
        if p_normal >= 0.2 or profile.guided_low_threshold is None:
            return seg
        low = seg.fg_prob > profile.guided_low_threshold
        if not low.any():
            return seg
        # Hysteresis: the lower threshold may only extend regions the model already
        # found; if it found none, keep the single strongest region.
        comp, n = ndimage.label(low)
        seeds = seg.labels > 0
        if seeds.any():
            keep = np.unique(comp[seeds & low])
            keep = keep[keep > 0]
        else:
            peak = np.asarray(ndimage.maximum(seg.fg_prob, comp, range(1, n + 1)))
            keep = np.array([int(np.argmax(peak)) + 1]) if peak.max() >= 0.35 else np.array([], dtype=int)
        mask = np.isin(comp, keep)
        if mask.sum() < MIN_TUMOR_PIXELS or mask.sum() <= (seg.labels > 0).sum():
            return seg
        labels = np.where(mask, seg.fg_label, 0).astype(np.uint8)
        p = np.clip(seg.fg_prob, 1e-6, 1 - 1e-6)
        binary = (-(p * np.log(p) + (1 - p) * np.log(1 - p)) / math.log(2)).astype(np.float32)
        uncertainty = np.where(mask & (seg.uncertainty == 0), binary, seg.uncertainty).astype(np.float32)
        note = "Outline threshold lowered because the classifier is confident a tumor is present; check the boundary."
        return Segmentation(seg.method, labels, uncertainty, seg.saliency, seg.fg_prob, seg.fg_label, profile.guided_low_threshold, note, seg.model)

    # ------------------------------------------------------------------ classification
    def key_slices(self, scan: Scan, seg: Segmentation, limit: int = 5) -> list[int]:
        areas = (seg.labels > 0).sum(axis=(1, 2))
        if areas.max() > 0:
            best = int(np.argmax(areas))
            half = limit // 2
            return [i for i in range(best - half, best + half + 1) if 0 <= i < len(areas) and areas[i] > 0]
        vol = normalize(scan.primary())
        fg = [i for i, s in enumerate(vol) if s.mean() > 0.02]
        if not fg:
            return [len(vol) // 2]
        return sorted({fg[int(round(j))] for j in np.linspace(0, len(fg) - 1, num=min(16, len(fg)))})

    def classify(self, scan: Scan, profile: CancerProfile, seg: Segmentation) -> tuple[Classification, np.ndarray]:
        """Classify the scan and return ``(classification, heatmap volume)``."""
        if profile.key not in self.classifiers:
            return self._heuristic_classify(scan, profile, seg)

        model, meta = self.classifiers[profile.key]
        classes = list(meta.get("classes", profile.classes))
        size = meta.get("image_size", profile.image_size)
        vol = normalize(scan.primary(profile.classifier_channel_preference))
        slices = self.key_slices(scan, seg)
        batch = torch.from_numpy(np.stack([resize2d(vol[i], (size, size)) for i in slices])[:, None].copy()).float()
        batch = batch.to(self.device)
        with torch.no_grad():
            enable_mc_dropout(model)
            samples = torch.stack([torch.softmax(model(batch), dim=1) for _ in range(self.mc_samples)])
            model.eval()
        per_sample = samples.mean(dim=1).cpu().numpy()  # (T, K) averaged over slices
        probs = per_sample.mean(axis=0)
        best = int(np.argmax(probs))
        label = classes[best]
        cls = Classification(
            method="model",
            label=label,
            display=profile.class_descriptions.get(label, label),
            probabilities={c: float(p) for c, p in zip(classes, probs)},
            confidence=float(probs[best]),
            uncertainty=float(_entropy(probs[None], axis=1)[0]),
            mc_std=float(per_sample[:, best].std()),
            slices_used=slices,
        )
        return cls, self._grad_cam_volume(model, vol, seg, slices, best, size)

    def _grad_cam_volume(self, model, vol, seg, key_slices, target, size) -> np.ndarray:
        z, h, w = vol.shape
        heat = np.zeros((z, h, w), dtype=np.float32)
        tumor = list(np.where(seg.labels.any(axis=(1, 2)))[0])
        chosen = sorted(set(tumor[:: max(1, math.ceil(len(tumor) / MAX_CAM_SLICES))]) | set(key_slices))
        for i in chosen:
            x = torch.from_numpy(resize2d(vol[i], (size, size))[None, None].copy()).float().to(self.device)
            cam = grad_cam(model, x, target).cpu().numpy()
            heat[i] = resize2d(cam, (h, w))
        return np.clip(heat, 0, 1)

    def _heuristic_classify(self, scan, profile, seg) -> tuple[Classification, np.ndarray]:
        saliency = seg.saliency
        if saliency is None:
            saliency = heuristic.segment(scan.primary(profile.classifier_channel_preference), profile)[2]
        score = heuristic.abnormality_score(seg.labels, saliency)
        abnormal = bool((seg.labels > 0).any())
        probs = {"abnormal": score, profile.normal_class: 1 - score}
        p = np.array([score, 1 - score])
        label = "abnormal" if abnormal else profile.normal_class
        display = (
            "Abnormal region detected (tumor type needs a trained classifier)"
            if abnormal
            else profile.class_descriptions[profile.normal_class]
        )
        cls = Classification(
            method="heuristic",
            label=label,
            display=display,
            probabilities=probs,
            confidence=float(max(score, 1 - score)),
            uncertainty=float(_entropy(p[None], axis=1)[0]),
            slices_used=self.key_slices(scan, seg),
        )
        return cls, saliency
