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
    longest, perpendicular = _diameters(core[best], dy, dx)
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
        location: dict = {"distance_from_midline": round(abs(offset), 1)}
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
        if profile.organ == "breast":
            location["depth_from_skin"] = round(float(np.nonzero(lesion[best])[0].min()) * dy, 1)
        result["location"] = location
    return plain(result)
