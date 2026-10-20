"""Point explanations for hovering over a scan or its 3D view.

Given a point (slice, row, column), say what is there in plain words: which tumor part
(and what that part means), or which area of the organ (approximately) and what it does,
plus how far the tumor is from it and what that could mean. One short line each.

Areas are estimated from the position inside the organ, not from an atlas, so they are
labeled "approx." Distances are in mm when the voxel size is known.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from ..imaging.lung import lung_mask
from ..imaging.measure import ONE_BREAST_FOV_MM
from ..imaging.preprocess import foreground_mask, normalize
from ..models.registry import get_profile

# What each tumor part is, and how it affects the patient or the tumor.
PARTS = {
    "edema": ("Edema: swollen brain tissue around the tumor (bright on FLAIR / T2).",
              "It adds pressure on nearby brain (mass effect) and often causes headaches; steroids can reduce it."),
    "enhancing_tumor": ("Active tumor: the growing rim where contrast leaks through damaged blood vessels.",
                        "This is the part doctors measure to judge whether treatment is working (RANO)."),
    "necrotic_core": ("Necrotic core: dead tumor tissue in the middle.",
                      "A sign of fast growth, typical of high-grade gliomas; it does not shrink much with treatment."),
    "tumor": ("Tumor (the model does not separate its parts here).",
              "Its size and position decide which nearby functions could be affected."),
    "lesion": ("Breast lesion seen on ultrasound.",
               "Its shape, edges and orientation help tell benign from malignant; a biopsy confirms."),
    "nodule": ("Lung nodule: a small round spot in the lung.",
               "Most are benign; its size and growth on follow-up CT decide what happens next (Lung-RADS)."),
    "possible_extent": ("Possible extent: how deep the tumor could reach.",
                        "One image cannot show depth; 9 out of 10 real tumors are shallower than this shell."),
}
BREAST_MRI_TUMOR = ("Breast tumor: tissue that takes up contrast quickly on MRI.",
                    "Its size is used for staging and to follow how it shrinks with chemotherapy.")

# One line about the tumor type the classifier suggests.
TYPES = {
    "glioma": "Glioma: grows from the brain's own support cells and can spread into nearby brain.",
    "meningioma": "Meningioma: grows from the brain's outer covering; usually slow-growing and benign, it pushes rather than invades.",
    "pituitary": "Pituitary tumor: at the hormone gland under the brain; usually benign, it can affect hormones and vision.",
    "benign": "Benign-looking lesion (e.g. fibroadenoma or cyst): does not spread; often only watched.",
    "malignant": "Malignant-looking features: needs a biopsy to confirm and decide treatment.",
}


def type_note(study: dict) -> str | None:
    cls = study["classification"]
    note = TYPES.get(cls["label"])
    return f"AI suggests ({cls['confidence']:.0%} confident): {note}" if note and cls["method"] == "model" else None


BRAIN_AREAS = {
    "frontal lobe": "planning, decisions, personality and, at its back edge, movement",
    "parietal lobe": "touch, body position and sense of space",
    "temporal lobe": "hearing, memory and, usually on the left, language",
    "occipital lobe": "vision",
    "cerebellum": "balance and coordination",
    "brainstem": "breathing, heart rate, swallowing and alertness",
    "deep brain (thalamus / basal ganglia)": "relaying signals and controlling movement",
}
BREAST_QUADRANTS = {
    "upper outer": "the most common place for breast cancer (about half start here)",
    "upper inner": "about 1 in 6 breast cancers start here",
    "lower outer": "about 1 in 10 breast cancers start here",
    "lower inner": "the least common place for breast cancer",
    "central (behind the nipple)": "milk ducts converge here",
}


@lru_cache(maxsize=2)  # a couple of studies; large CT arrays are hundreds of MB each
def _context(sid: str, version: str):
    """Arrays and derived maps for one study, cached across hover requests."""
    from .pipeline import get_store  # late import: pipeline imports this module's users

    arrays = get_store().arrays(sid)
    base = normalize(next(v for k, v in arrays.items() if k.startswith("ch_")))
    labels = arrays["labels"]
    return arrays, base, labels


@lru_cache(maxsize=2)  # a couple of studies; large CT arrays are hundreds of MB each
def _distance_map(sid: str, version: str, spacing: tuple[float, float, float] | None):
    """Distance (mm or px) from every voxel to the nearest tumor voxel, at up to 256 px in-plane."""
    _, _, labels = _context(sid, version)
    tumor = labels > 0
    if not tumor.any():
        return None, 1
    step = max(1, int(np.ceil(max(tumor.shape[1:]) / 256)))
    small = tumor[:, ::step, ::step]
    sp = spacing or (1.0, 1.0, 1.0)
    if small.shape[0] == 1:
        dist = ndimage.distance_transform_edt(~small[0], sampling=(sp[1] * step, sp[2] * step))[None]
    else:
        dist = ndimage.distance_transform_edt(~small, sampling=(sp[0], sp[1] * step, sp[2] * step))
    return dist.astype(np.float32), step


@lru_cache(maxsize=2)  # a couple of studies; large CT arrays are hundreds of MB each
def _organ(sid: str, version: str, organ: str):
    """Organ mask and its bounding box (slices, rows, cols), for locating areas."""
    _, base, _ = _context(sid, version)
    if organ == "lung" and base.shape[0] > 1:
        mask = lung_mask(base)
    else:
        mask = np.stack([ndimage.binary_fill_holes(foreground_mask(s)) for s in base])
    if not mask.any():
        return mask, None
    box = [np.where(mask.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    return mask, [(int(b[0]), int(b[-1])) for b in box]


def _frac(v: float, lo_hi: tuple[int, int]) -> float:
    lo, hi = lo_hi
    return float(np.clip((v - lo) / max(hi - lo, 1), 0, 1))


def _brain_area(z: float, r: float, c: float, box, is_3d: bool) -> str:
    ap = _frac(r, box[1])  # 0 = front (anterior), 1 = back
    lateral = abs(_frac(c, box[2]) - 0.5) * 2  # 0 = midline, 1 = side
    height = _frac(z, box[0]) if is_3d else 0.6  # 0 = bottom (inferior), 1 = top
    if height < 0.3 and ap > 0.6:
        return "cerebellum"
    if height < 0.35 and lateral < 0.25 and 0.35 < ap <= 0.65:
        return "brainstem"
    if ap < 0.38:
        return "frontal lobe"
    if ap > 0.78:
        return "occipital lobe"
    if lateral > 0.55 and height < 0.6:
        return "temporal lobe"
    if lateral < 0.3 and 0.35 < height < 0.6:
        return "deep brain (thalamus / basal ganglia)"
    return "parietal lobe"


def _area(profile, z: float, r: float, c: float, box, is_3d: bool, width_mm: float | None = None) -> tuple[str, str]:
    """(area name, what it does) for a point inside the organ."""
    if box is None:
        return "this area", ""
    mid = (box[2][0] + box[2][1]) / 2
    side = "right" if c < mid else "left"  # radiological convention: image left = patient right
    if profile.organ == "brain":
        if not is_3d:
            # One image of unknown direction (axial, coronal or sagittal): lobes cannot be named reliably.
            vertical = "upper" if _frac(r, box[1]) < 0.5 else "lower"
            return (f"brain tissue, {vertical} {'left' if c < mid else 'right'} part of the image",
                    "one image cannot show which lobe this is (a 3D scan can)")
        name = _brain_area(z, r, c, box, is_3d)
        if name in ("brainstem", "deep brain (thalamus / basal ganglia)"):
            return f"{name} (approx.)", BRAIN_AREAS[name]
        return f"{side} {name} (approx.)", BRAIN_AREAS[name]
    if profile.organ == "lung":
        if not is_3d:
            return f"{side} lung", "air exchange: oxygen in, carbon dioxide out"
        level = 1 - _frac(z, box[0])  # 0 = top (apex)
        zone = "upper" if level < 0.34 else "lower" if level > 0.66 else "middle"
        extra = "; most smoking-related lung cancers start in the upper zones" if zone == "upper" else ""
        return f"{side} lung, {zone} zone (approx.)", "air exchange: oxygen in, carbon dioxide out" + extra
    if profile.modality == "Ultrasound":
        return "breast tissue in the ultrasound image", "the image shows a slice from the skin (top) inwards"
    if width_mm is not None and width_mm < ONE_BREAST_FOV_MM:  # only one breast in the scan
        return "breast tissue (one breast in this scan; side not known)", "breast tissue: milk glands, ducts and fat"
    # Breast MRI: which breast, and quadrant from position within that breast.
    half = (box[2][0], mid) if side == "right" else (mid, box[2][1])
    outer = abs(_frac(c, half) - (0.0 if side == "right" else 1.0)) < 0.4  # far from the midline
    if not is_3d:  # an axial image shows front-back, not up-down, so only the half is known
        return f"{side} breast, {'outer' if outer else 'inner'} half (approx.)", "breast tissue: milk glands, ducts and fat"
    vertical = "upper" if _frac(z, box[0]) > 0.5 else "lower"
    q = f"{vertical} {'outer' if outer else 'inner'}"
    return f"{side} breast, {q} quadrant (approx.)", BREAST_QUADRANTS[q]


def explain_point(study: dict, version: str, z: int, r: int, c: int) -> dict:
    """Plain-language explanation of what is at voxel (z, r, c) of a stored study."""
    profile = get_profile(study["profile"]["key"])
    arrays, base, labels = _context(study["id"], version)
    zmax, rmax, cmax = labels.shape
    if not (0 <= z < zmax and 0 <= r < rmax and 0 <= c < cmax):
        return {"inside": False}
    spacing = tuple(study["scan"]["spacing_mm"]) if study["scan"]["spacing_mm"] else None
    units = "mm" if spacing else "px"
    is_3d = zmax > 1
    detection = "tumor" in study["classification"]["probabilities"]
    heat = float(arrays["heat"][z, r, c])
    out: dict = {"inside": True, "slice": z, "row": r, "col": c}
    if detection:
        out["probability"] = round(heat, 2)  # the outline model's tumor probability
    else:
        out["attention"] = round(heat, 2)  # Grad-CAM: how much this spot drove the classifier

    value = int(labels[z, r, c])
    if value:
        lbl = profile.label(value)
        what, effect = BREAST_MRI_TUMOR if profile.key == "breast_mri" else PARTS.get(lbl.name, (lbl.description, ""))
        out.update({"kind": "tumor", "title": lbl.description, "what": what, "effect": effect, "color": list(lbl.color), "type_note": type_note(study)})
        region = next((x for x in study["measurements"].get("regions", []) if x["name"] == lbl.name), None)
        if region:
            size = region.get("volume_cm3", region.get("area_mm2"))
            size = round(size, 1) if size is not None else None
            out["size"] = f"{size} {'cm³' if 'volume_cm3' in region else 'mm²'} ({region['percent_of_tumor']}% of the tumor)" if size is not None else None
        nodules = study["measurements"].get("nodules")
        if nodules:
            n = min(nodules, key=lambda n: (n["centroid"]["slice"] - z) ** 2 + (n["centroid"]["row"] - r) ** 2 + (n["centroid"]["col"] - c) ** 2)
            from ..imaging.measure import lung_rads

            out["size"] = f"Nodule {nodules.index(n) + 1}: {n['longest_diameter']} x {n['short_diameter']} {units}"
            if spacing:  # Lung-RADS needs sizes in mm, not pixels
                out["size"] += f", size-based Lung-RADS {lung_rads(n['mean_diameter'])['category']}"
        return out

    organ, box = _organ(study["id"], version, profile.organ)
    if not organ[z, r, c] and profile.organ != "brain":
        if profile.organ == "lung":
            out.update({"kind": "outside", "title": "Outside the lungs", "what": "Chest wall, heart, blood vessels or other tissue.", "effect": ""})
            return out
    if not organ[z, r, c] and profile.organ == "brain":
        out.update({"kind": "outside", "title": "Outside the brain", "what": "Skull, scalp or background.", "effect": ""})
        return out
    width_mm = cmax * spacing[2] if spacing else None
    name, function = _area(profile, z, r, c, box, is_3d, width_mm)
    lobe = profile.organ == "brain" and is_3d
    out.update({"kind": "area", "title": name[0].upper() + name[1:], "what": f"Responsible for {function}." if function and lobe else (function[0].upper() + function[1:] + "." if function else "")})
    dist, step = _distance_map(study["id"], version, spacing)
    if dist is None:
        out["effect"] = "No tumor was outlined, so nothing here is affected by one."
        return out
    d = float(dist[min(z, dist.shape[0] - 1), min(r // step, dist.shape[1] - 1), min(c // step, dist.shape[2] - 1)])
    out["distance"] = round(d, 1)
    affects = f"could affect {function.split(';')[0]}" if function and lobe else "could press on this tissue"
    if d <= 3:
        out["effect"] = f"The tumor reaches this area, so it {affects}."
    elif d <= 15:
        out["effect"] = f"The tumor is {d:.0f} {units} away; its growth or swelling {affects}."
    else:
        out["effect"] = f"The tumor is {d:.0f} {units} away; this area is unlikely to be directly affected."
    return out
