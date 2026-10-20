"""Quantitative measurements of a segmented tumor.

Measurements follow the conventions radiologists use for response assessment:
the longest in-plane diameter on the slice where the tumor is largest, the
longest diameter perpendicular to it, and their product (as in RANO / WHO
criteria), plus volume per tumor sub-region.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ..models.registry import CancerProfile
from ..utils import plain
from .preprocess import foreground_mask, normalize


# Breast MRI narrower than this shows only one breast (both breasts need about 300 mm or more).
ONE_BREAST_FOV_MM = 280


def _diameters(mask2d: np.ndarray, dy: float, dx: float) -> tuple[float, float]:
    """Longest diameter and longest perpendicular diameter of a 2D mask, in mm (or px)."""
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return 0.0, 0.0
    pts = np.column_stack([ys * dy, xs * dx]).astype(np.float64)
    pixel = (dy + dx) / 2
    if ys.size < 3:
        return pixel, pixel
    try:
        hull = pts[ConvexHull(pts).vertices]
    except QhullError:  # collinear points
        hull = pts
    diff = hull[:, None, :] - hull[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    longest = float(dist[i, j])
    if longest == 0:
        return pixel, pixel
    axis = (hull[j] - hull[i]) / longest
    perp = np.array([-axis[1], axis[0]])
    proj = pts @ perp
    return longest + pixel, float(proj.max() - proj.min()) + pixel


def _brain_region(row_frac: float, col_frac: float, z_frac: float | None) -> str:
    """Very rough lobe estimate from position in an axial head slice."""
    if z_frac is not None and z_frac < 0.25 and 0.35 < col_frac < 0.65 and 0.35 < row_frac < 0.6:
        return "skull base / sellar region"
    if row_frac < 0.35:
        return "frontal region"
    if row_frac > 0.72:
        return "occipital region"
    if col_frac < 0.3 or col_frac > 0.7:
        return "temporal region"
    return "parietal / deep central region"


def measure(
    labels: np.ndarray,
    reference: np.ndarray,
    profile: CancerProfile,
    spacing: tuple[float, float, float] | None,
    axial_volume: bool,
) -> dict:
    """Measure the tumor in ``labels`` (Z, H, W). ``reference`` is the image it was found on."""
    lesion = labels > 0
    units = "mm" if spacing else "px"
    dz, dy, dx = spacing if spacing else (1.0, 1.0, 1.0)
    voxel = dz * dy * dx
    is_3d = labels.shape[0] > 1
    result: dict = {"detected": bool(lesion.any()), "units": units, "is_3d": is_3d}
    if not lesion.any():
        return result

    def size_of(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if is_3d:
            return {"voxels": n, "volume_cm3": round(n * voxel / 1000, 3)} if spacing else {"voxels": n}
        area = n * dy * dx
        return {"pixels": n, "area_mm2": round(area, 1), "area_cm2": round(area / 100, 3)} if spacing else {"pixels": n}

    total = size_of(lesion)
    result["total"] = total
    regions = []
    for lbl in profile.display_labels:
        mask = labels == lbl.index
        if mask.any():
            regions.append(
                {"name": lbl.name, "description": lbl.description, **size_of(mask), "percent_of_tumor": round(100 * mask.sum() / lesion.sum(), 1)}
            )
    result["regions"] = regions

    _, n_components = ndimage.label(lesion)
    result["lesion_count"] = int(n_components)

    # RANO measures the contrast-enhancing core; fall back to the whole lesion.
    core = np.isin(labels, (1, 3)) if profile.organ == "brain" else lesion
    if not core.any():
        core = lesion
    areas = core.sum(axis=(1, 2))
    best = int(np.argmax(areas))
    # Diameters are measured on the largest single lesion, as in RANO / RECIST.
    comp, n = ndimage.label(core[best])
    main = comp == (int(np.argmax(ndimage.sum(core[best], comp, range(1, n + 1)))) + 1) if n > 1 else core[best]
    longest, perpendicular = _diameters(main, dy, dx)
    result["key_slice"] = best
    result["longest_diameter"] = round(longest, 1)
    result["perpendicular_diameter"] = round(perpendicular, 1)
    result["bidimensional_product"] = round(longest * perpendicular, 1)

    zs = np.where(lesion.any(axis=(1, 2)))[0]
    result["slice_range"] = [int(zs.min()), int(zs.max())]
    if is_3d:
        result["craniocaudal_extent"] = round((zs.max() - zs.min() + 1) * dz, 1)

    # Location relative to the imaged body part on the key slice.
    cz, cy, cx = ndimage.center_of_mass(lesion)
    result["centroid"] = {"slice": round(cz, 1), "row": round(cy, 1), "col": round(cx, 1)}
    fg = foreground_mask(normalize(reference[best]))
    if fg.any():
        rows, cols = np.where(fg.any(axis=1))[0], np.where(fg.any(axis=0))[0]
        midline = (cols[0] + cols[-1]) / 2
        row_frac = (cy - rows[0]) / max(rows[-1] - rows[0], 1)
        col_frac = (cx - cols[0]) / max(cols[-1] - cols[0], 1)
        offset = (cx - midline) * dx
        location: dict = {}
        # Breast MRI showing only one breast (many ISPY scans): its centre is not the body's midline.
        one_breast = profile.organ == "breast" and profile.modality == "MRI" and labels.shape[2] * dx < ONE_BREAST_FOV_MM
        if one_breast:
            location["field_of_view"] = "one breast only (left or right cannot be told from this scan)"
        elif profile.modality != "Ultrasound":  # an ultrasound image has no patient midline
            location["distance_from_midline"] = round(abs(offset), 1)
            if abs(offset) < 0.04 * (cols[-1] - cols[0]) * dx:
                location["side"] = "midline"
            else:
                # Radiological convention: image left = patient right.
                location["side"] = "patient right" if offset < 0 else "patient left"
        vertical = "upper" if row_frac < 0.4 else "lower" if row_frac > 0.6 else "middle"
        horizontal = "left" if col_frac < 0.4 else "right" if col_frac > 0.6 else "central"
        location["image_position"] = f"{vertical} {horizontal} part of the image"
        if profile.organ == "brain" and axial_volume:
            z_frac = cz / max(labels.shape[0] - 1, 1) if is_3d else None
            location["approximate_region"] = _brain_region(row_frac, col_frac, z_frac)
        if profile.modality == "Ultrasound":
            location["depth_from_skin"] = round(float(np.nonzero(lesion[best])[0].min()) * dy, 1)
        result["location"] = location
    if profile.organ == "lung" and is_3d:
        result["nodules"] = _nodules(lesion, reference, (dz, dy, dx), bool(spacing))
        if spacing and result["nodules"]:
            result["lung_rads"] = lung_rads(max(n["mean_diameter"] for n in result["nodules"]))
    return plain(result)


def _nodules(lesion: np.ndarray, reference: np.ndarray, spacing, has_spacing: bool, limit: int = 10) -> list[dict]:
    """Each nodule's size and position, largest first."""
    dz, dy, dx = spacing
    comp, n = ndimage.label(lesion)
    if not n:
        return []
    sizes = ndimage.sum(lesion, comp, range(1, n + 1))
    body = np.stack([foreground_mask(s) for s in normalize(reference)[:: max(1, reference.shape[0] // 16)]]).any(0)
    cols = np.where(body.any(axis=0))[0]
    midline = (cols[0] + cols[-1]) / 2 if cols.size else lesion.shape[2] / 2
    out = []
    for i in np.argsort(sizes)[::-1][:limit]:
        mask = comp == i + 1
        areas = mask.sum(axis=(1, 2))
        z = int(np.argmax(areas))
        longest, short = _diameters(mask[z], dy, dx)
        cz, cy, cx = ndimage.center_of_mass(mask)
        nodule = {
            "key_slice": z,
            "longest_diameter": round(longest, 1),
            "short_diameter": round(short, 1),
            "mean_diameter": round((longest + short) / 2, 1),
            "volume_mm3": round(float(sizes[i]) * dz * dy * dx, 1) if has_spacing else None,
            # Radiological convention: image left = patient right.
            "side": "right lung" if cx < midline else "left lung",
            "centroid": {"slice": round(cz, 1), "row": round(cy, 1), "col": round(cx, 1)},
        }
        out.append(nodule)
    return out


def lung_rads(mean_diameter_mm: float) -> dict:
    """Lung-RADS v2022 category from the largest nodule's mean diameter, assuming a solid nodule on a
    first (baseline) screening CT. The real category also depends on nodule type (solid, part-solid,
    ground glass), growth on earlier scans and suspicious features."""
    d = mean_diameter_mm
    if d < 6:
        category, meaning, follow_up = "2", "Benign appearance or behavior", "Continue annual screening CT in 12 months"
    elif d < 8:
        category, meaning, follow_up = "3", "Probably benign", "Follow-up low-dose CT in 6 months"
    elif d < 15:
        category, meaning, follow_up = "4A", "Suspicious", "Low-dose CT in 3 months; PET/CT may be considered"
    else:
        category, meaning, follow_up = "4B", "Very suspicious", "Chest CT, PET/CT and/or tissue sampling, per the care team"
    return {
        "category": category,
        "meaning": meaning,
        "typical_management": follow_up,
        "based_on_mean_diameter_mm": round(d, 1),
        "assumptions": "Solid nodule on a baseline screening CT. The real category also depends on nodule type, "
        "growth compared with earlier scans and other features that this tool does not assess.",
    }
