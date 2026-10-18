"""Comparing studies of the same patient over time.

Response categories use simplified imaging-only versions of standard criteria:
RANO for brain tumors (bidimensional product of the enhancing core) and
RECIST 1.1 for other solid tumors (longest diameter). Real RANO/RECIST also
consider new lesions, steroid dose, clinical status and non-target lesions, so
these labels are a guide for discussion only.
"""

from __future__ import annotations


def _size(summary: dict) -> float | None:
    m = summary.get("measurements", {})
    if not m.get("detected"):
        return 0.0
    total = m.get("total", {})
    return total.get("volume_cm3") or total.get("area_mm2") or total.get("voxels") or total.get("pixels")


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return round(100.0 * (new - old) / old, 1)


def response_category(organ: str, prior: dict, current: dict) -> dict:
    pm, cm = prior.get("measurements", {}), current.get("measurements", {})
    if organ == "brain":
        metric, before, after = "bidimensional product", pm.get("bidimensional_product", 0.0), cm.get("bidimensional_product", 0.0)
        thresholds, criteria = (-50.0, 25.0), "RANO (simplified, imaging only)"
    else:
        metric, before, after = "longest diameter", pm.get("longest_diameter", 0.0), cm.get("longest_diameter", 0.0)
        thresholds, criteria = (-30.0, 20.0), "RECIST 1.1 (simplified, single target lesion)"
    before, after = before or 0.0, after or 0.0
    change = _pct(after, before)
    if before == 0 and after == 0:
        category = "no measurable disease"
    elif before == 0:
        category = "new lesion"
    elif after == 0:
        category = "complete response"
    elif change <= thresholds[0]:
        category = "partial response"
    elif change >= thresholds[1]:
        category = "progressive disease"
    else:
        category = "stable disease"
    return {"criteria": criteria, "metric": metric, "before": before, "after": after, "change_percent": change, "category": category}


def compare(prior: dict, current: dict) -> dict:
    if prior["profile"]["key"] != current["profile"]["key"]:
        raise ValueError("Can only compare studies of the same scan type.")
    if prior["scan_date"] > current["scan_date"]:
        prior, current = current, prior
    size_before, size_after = _size(prior), _size(current)
    pm, cm = prior["measurements"], current["measurements"]
    unit = "cm³" if current["measurements"].get("is_3d") else "mm²"
    if current["measurements"].get("units") == "px" or prior["measurements"].get("units") == "px":
        unit = "voxels" if current["measurements"].get("is_3d") else "pixels"
    regions = {}
    for study, key in ((prior, "before"), (current, "after")):
        for r in study["measurements"].get("regions", []):
            regions.setdefault(r["name"], {"before": 0.0, "after": 0.0})
            regions[r["name"]][key] = r.get("volume_cm3") or r.get("area_mm2") or r.get("voxels") or r.get("pixels") or 0.0
    for r in regions.values():
        r["change_percent"] = _pct(r["after"], r["before"])
    from datetime import date

    days = (date.fromisoformat(current["scan_date"][:10]) - date.fromisoformat(prior["scan_date"][:10])).days
    return {
        "prior_study": prior["id"],
        "current_study": current["id"],
        "prior_date": prior["scan_date"],
        "current_date": current["scan_date"],
        "interval_days": days,
        "size_unit": unit,
        "size_before": size_before,
        "size_after": size_after,
        "size_change_percent": _pct(size_after, size_before),
        "longest_diameter_before": pm.get("longest_diameter"),
        "longest_diameter_after": cm.get("longest_diameter"),
        "regions": regions,
        "response": response_category(current["profile"]["organ"], prior, current),
        "spacing_estimated": bool(prior["scan"]["spacing_estimated"] or current["scan"]["spacing_estimated"]),
    }


def timeline(studies: list[dict]) -> list[dict]:
    """Size over time for a patient's studies (all scan types), oldest first."""
    points = []
    for s in sorted(studies, key=lambda s: (s["scan_date"], s["created_at"])):
        m = s["measurements"]
        points.append(
            {
                "study_id": s["id"],
                "scan_date": s["scan_date"],
                "profile": s["profile"]["key"],
                "size": _size(s),
                "longest_diameter": m.get("longest_diameter", 0.0),
                "bidimensional_product": m.get("bidimensional_product", 0.0),
                "label": s["classification"]["label"],
            }
        )
    return points
