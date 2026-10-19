"""End-to-end analysis of one uploaded study."""

from __future__ import annotations

import threading
from datetime import date

import numpy as np

from ..config import settings
from ..imaging.io import Scan, load_scan
from ..imaging.measure import measure
from ..imaging.preprocess import estimate_brain_spacing, normalize
from ..models.engine import Engine, RouteResult
from ..models.registry import PROFILES, get_profile
from ..storage import Store, now_iso
from ..utils import plain

DISCLAIMER = (
    "OncoVision is a research and education tool. It is not a medical device and its output "
    "is not a diagnosis. All findings must be reviewed by a qualified radiologist or physician."
)

_engine: Engine | None = None
_store: Store | None = None
_init_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    with _init_lock:
        if _engine is None:
            _engine = Engine(settings.weights_dir, settings.device, settings.mc_samples)
    return _engine


def get_store() -> Store:
    global _store
    with _init_lock:
        if _store is None:
            _store = Store(settings.data_dir)
    return _store


class UnsupportedScanError(ValueError):
    pass


def apply_window(scan: Scan, profile) -> None:
    """Clip CT values in Hounsfield units to the profile's window (e.g. lung window)."""
    if not profile.intensity_window or scan.source_format not in ("dicom", "nifti"):
        return
    lo, hi = profile.intensity_window
    for name, arr in scan.channels.items():
        if arr.min() < -500:  # looks like HU; plain-intensity data is left alone
            scan.channels[name] = np.clip(arr, lo, hi)


def _review_reasons(route: RouteResult, cls, seg, measurements: dict, lesion_uncertainty: float, profile) -> list[str]:
    reasons = []
    if route.method != "user" and route.confidence < 0.7:
        reasons.append(f"Scan type was guessed with low confidence ({route.confidence:.0%}); please confirm it.")
    if cls.method == "heuristic" or seg.method == "heuristic":
        reasons.append("Demo mode: at least one step used the heuristic detector instead of a trained model.")
    if cls.confidence < settings.min_confidence or (cls.mc_std or 0.0) > settings.max_mc_std:
        reasons.append(f"Classifier is unsure (confidence {cls.confidence:.0%}).")
    if lesion_uncertainty > (settings.review_threshold or profile.review_uncertainty):
        reasons.append(f"Tumor boundary is uncertain (mean uncertainty {lesion_uncertainty:.2f}).")
    detected = measurements.get("detected", False)
    says_normal = cls.label == profile.normal_class
    if detected and says_normal and cls.method == "model":
        reasons.append("Models disagree: a region was segmented but the classifier predicts no tumor.")
    if not detected and not says_normal and cls.method == "model":
        reasons.append(
            f"The classifier found {cls.label.replace('_', ' ')}, but the outline model could not delineate it on this image "
            "(it was trained mostly on contrast-enhanced T1 slices), so there are no size measurements. The AI heatmap "
            "shows the approximate location."
        )
    return reasons


def analyze(
    files: list[tuple[str, bytes, str | None]],
    scan_type: str = "auto",
    patient_label: str = "",
    scan_date: str | None = None,
    pixel_spacing_mm: float | None = None,
) -> dict:
    engine, store = get_engine(), get_store()
    scan: Scan = load_scan(files)

    if pixel_spacing_mm:
        dz = scan.spacing[0] if scan.spacing else pixel_spacing_mm
        scan.spacing = (dz, pixel_spacing_mm, pixel_spacing_mm)
        scan.spacing_estimated = False

    if scan_type and scan_type != "auto":
        route = RouteResult(get_profile(scan_type).key, 1.0, "user", reason="Selected by the user.")
    else:
        route = engine.route(scan)
    if route.profile and route.method != "user" and not engine.demo_mode and not engine.has_models(route.profile):
        raise UnsupportedScanError(
            f"This looks like a {get_profile(route.profile).display_name} scan, but OncoVision has no trained model for it "
            "yet, so it cannot give a reliable result. Choose the scan type manually to run the experimental demo detector."
        )
    if route.profile is None:
        supported = [p.display_name for p in PROFILES.values() if engine.demo_mode or engine.has_models(p.key)]
        raise UnsupportedScanError(
            f"This scan does not look like one of the supported types ({', '.join(supported)}). "
            "Pick the scan type manually to analyze anyway."
        )
    profile = get_profile(route.profile)
    apply_window(scan, profile)

    if scan.spacing is None and profile.organ == "brain":
        est = estimate_brain_spacing(scan.primary())
        if est:
            scan.spacing = (est, est, est)
            scan.spacing_estimated = True

    seg = engine.segment(scan, profile)
    cls, heat = engine.classify(scan, profile, seg)
    seg = engine.refine(seg, cls, profile)
    reference = scan.primary(profile.classifier_channel_preference)
    measurements = measure(
        seg.labels, reference, profile, scan.spacing, axial_volume=scan.source_format in ("nifti", "dicom")
    )
    lesion = seg.labels > 0
    lesion_unc = float(seg.uncertainty[lesion].mean()) if lesion.any() else 0.0
    reasons = _review_reasons(route, cls, seg, measurements, lesion_unc, profile)

    patient = store.patient_for_label(patient_label)
    methods = {cls.method, seg.method}
    summary = {
        "id": store.new_study_id(),
        "patient": {"id": patient["id"], "label": patient["label"]},
        "scan_date": scan_date or date.today().isoformat(),
        "created_at": now_iso(),
        "mode": "demo" if methods == {"heuristic"} else "model" if methods == {"model"} else "mixed",
        "profile": {
            "key": profile.key,
            "display_name": profile.display_name,
            "organ": profile.organ,
            "modality": profile.modality,
            "reporting_system": profile.reporting_system,
            "seg_labels": [
                {"index": l.index, "name": l.name, "description": l.description, "color": l.color} for l in profile.display_labels
            ],
        },
        "route": {"method": route.method, "confidence": route.confidence, "reason": route.reason, "probabilities": route.probabilities},
        "scan": {
            "shape": list(scan.shape),
            "is_volume": scan.is_volume,
            "sequences": list(scan.channels),
            "source_format": scan.source_format,
            "spacing_mm": list(scan.spacing) if scan.spacing else None,
            "spacing_estimated": scan.spacing_estimated,
            "orientation_assumed": scan.orientation_assumed,
            "metadata": scan.metadata,
            "removed_phi": scan.removed_phi,
        },
        "classification": {
            "method": cls.method,
            "label": cls.label,
            "display": cls.display,
            "probabilities": cls.probabilities,
            "confidence": cls.confidence,
            "uncertainty": cls.uncertainty,
            "mc_std": cls.mc_std,
            "slices_used": cls.slices_used,
        },
        "segmentation": {"method": seg.method, "mean_uncertainty_in_tumor": lesion_unc, "threshold": seg.threshold, "note": seg.note},
        "measurements": measurements,
        "review": {"required": bool(reasons), "reasons": reasons},
        "disclaimer": DISCLAIMER,
    }
    summary = plain(summary)
    channels = {name: normalize(arr) for name, arr in scan.channels.items()}
    store.save_study(summary, channels, seg.labels, heat.astype(np.float32), seg.uncertainty)
    return summary
