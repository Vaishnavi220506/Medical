"""3D surface meshes (marching cubes) for the browser 3D viewer."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from ..models.registry import CancerProfile
from .preprocess import foreground_mask, normalize

MAX_DIM = 128  # downsample large volumes so meshes stay light in the browser


def _surface(mask: np.ndarray, spacing, step: int = 1, sigma: float = 1.0) -> dict | None:
    if mask.sum() < 8:
        return None
    padded = np.pad(mask.astype(np.float32), 1)
    padded = ndimage.gaussian_filter(padded, sigma)  # smooth, organic-looking surfaces
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing, step_size=step)
    except (ValueError, RuntimeError):
        return None
    verts -= np.array(spacing)  # undo padding offset
    return {"vertices": np.round(verts, 2).ravel().tolist(), "faces": faces.astype(int).ravel().tolist()}


def build_meshes(volume: np.ndarray, labels: np.ndarray, profile: CancerProfile, spacing) -> dict:
    """Meshes of the body outline (head) and each tumor sub-region, in mm (z, y, x order)."""
    spacing = tuple(spacing) if spacing else (1.0, 1.0, 1.0)
    factor = max(1, int(np.ceil(max(volume.shape) / MAX_DIM)))
    vol = normalize(volume[::factor, ::factor, ::factor] if factor > 1 else volume)
    lab = labels[::factor, ::factor, ::factor] if factor > 1 else labels
    sp = tuple(s * factor for s in spacing)

    body = np.stack([foreground_mask(s) for s in vol])
    out: dict = {"spacing": sp, "voxel_mm": list(spacing), "shape": list(volume.shape), "meshes": []}
    surface = _surface(ndimage.binary_fill_holes(body), sp, step=2, sigma=1.6)
    if surface:
        out["meshes"].append({"name": "anatomy", "color": [200, 200, 210], "opacity": 0.12, **surface})
    for lbl in profile.display_labels:
        surface = _surface(lab == lbl.index, sp)
        if surface:
            out["meshes"].append({"name": lbl.name, "description": lbl.description, "color": list(lbl.color), "opacity": 0.9, **surface})
    return out


PSEUDO_SIZE = 128  # working resolution for the estimated 3D preview


def _inflate(mask2d: np.ndarray, depth: int, flatten: float = 1.0) -> np.ndarray:
    """Turn a 2D region into a rounded 3D shape (Z, H, W).

    Each pixel gets a half-thickness of sqrt(d * (2R - d)), where d is its distance to the
    region's edge and R the largest such distance in that region: an exact sphere for a disk, and a
    smooth rounded body for other outlines.
    """
    d = ndimage.distance_transform_edt(mask2d)
    # R is taken per connected region, so a small spot stays small next to a large one.
    comp, n = ndimage.label(mask2d)
    r = np.zeros(n + 1)
    if n:
        r[1:] = ndimage.maximum(d, comp, np.arange(1, n + 1))
    half = np.sqrt(np.clip(d * (2 * r[comp] - d), 0, None)) * flatten
    z = np.abs(np.arange(depth) - (depth - 1) / 2)
    return z[:, None, None] <= half[None]


def build_pseudo_meshes(volume: np.ndarray, labels: np.ndarray, profile: CancerProfile, spacing=None) -> dict:
    """Estimated 3D preview from ONE 2D image, for visualization only.

    The real depth of the tumor is unknown from a single slice, so each outline is
    inflated into a rounded shape. Nothing here is a measurement: the API marks the
    result as pseudo and the viewer shows no volumes.
    """
    if volume.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Pseudo-3D preview expects one image slice")
    img, lab = normalize(volume[0]), labels[0]
    factor = max(1, int(np.ceil(max(img.shape) / PSEUDO_SIZE)))
    img, lab = img[::factor, ::factor], lab[::factor, ::factor]
    pixel = (spacing[1] if spacing else 1.0) * factor
    sp = (pixel, pixel, pixel)

    body = ndimage.binary_fill_holes(foreground_mask(img))
    r_body = ndimage.distance_transform_edt(body).max() if body.any() else 1.0
    depth = int(2 * r_body * 0.8) + 6
    out: dict = {"spacing": sp, "voxel_mm": list(sp), "shape": [img.shape[0], depth, img.shape[1]], "meshes": [], "pseudo": True}

    def upright(vol3d):
        # (depth, row, col) -> (row from the bottom, depth, col): the viewer's vertical axis is
        # the first one, so the image stands upright and the Front view shows it as uploaded.
        return np.ascontiguousarray(vol3d.transpose(1, 0, 2)[::-1])

    surface = _surface(upright(_inflate(body, depth, flatten=0.8)), sp, step=2, sigma=1.6)
    if surface:
        out["meshes"].append({"name": "anatomy", "color": [200, 200, 210], "opacity": 0.12, **surface})

    # Nested tumor parts are inflated from the outside in (whole tumor, core, necrosis),
    # so inner parts sit inside outer ones as in the real anatomy.
    present = set(np.unique(lab)) - {0}
    if profile.organ == "brain" and present & {1, 2, 3}:
        levels = [(lab > 0, 2 if 2 in present else 3), (np.isin(lab, (1, 3)), 3), (lab == 1, 1)]
    else:
        levels = [(lab == v, int(v)) for v in sorted(present)]
    for mask, value in levels:
        if mask.sum() < 4:
            continue
        lbl = profile.label(value)
        surface = _surface(upright(_inflate(mask, depth)), sp, sigma=1.0)
        if surface:
            out["meshes"].append({"name": lbl.name, "description": lbl.description, "color": list(lbl.color), "opacity": 0.9, **surface})
    out["warning"] = "The depth is estimated from a single image, so the shape and size are not measurements."
    return out
