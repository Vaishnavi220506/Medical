"""3D surface meshes (marching cubes) for the browser 3D viewer."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from ..models.registry import CancerProfile
from .preprocess import foreground_mask, normalize

MAX_DIM = 128  # downsample large volumes so meshes stay light in the browser


def _surface(mask: np.ndarray, spacing, step: int = 1) -> dict | None:
    if mask.sum() < 8:
        return None
    padded = np.pad(mask.astype(np.float32), 1)
    padded = ndimage.gaussian_filter(padded, 0.8)
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
    out: dict = {"spacing": sp, "shape": list(vol.shape), "meshes": []}
    surface = _surface(body, sp, step=2)
    if surface:
        out["meshes"].append({"name": "anatomy", "color": [200, 200, 210], "opacity": 0.12, **surface})
    for lbl in profile.display_labels:
        surface = _surface(lab == lbl.index, sp)
        if surface:
            out["meshes"].append({"name": lbl.name, "description": lbl.description, "color": list(lbl.color), "opacity": 0.9, **surface})
    return out


def build_pseudo_meshes(volume: np.ndarray, labels: np.ndarray, profile: CancerProfile) -> dict:
    """Build a visualization-only estimated extrusion from one 2D slice."""
    if volume.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Pseudo-3D preview expects one image slice")
    depth = 16
    label2d = labels[0]
    pseudo_volume = np.repeat(volume, depth, axis=0)
    pseudo_labels = np.zeros((depth, *label2d.shape), dtype=labels.dtype)
    taper = np.sin(np.linspace(0, np.pi, depth))
    for z, amount in enumerate(taper):
        for lbl in profile.display_labels:
            region = label2d == lbl.index
            if not region.any():
                continue
            distance = ndimage.distance_transform_edt(region)
            threshold = (1.0 - amount) * max(1.0, float(distance.max()) * 0.8)
            pseudo_labels[z][distance >= threshold] = lbl.index
    out = build_meshes(pseudo_volume, pseudo_labels, profile, (1.0, 1.0, 1.0))
    out["pseudo"] = True
    out["warning"] = "Estimated pseudo-3D visualization from one 2D image; depth and volume are not clinical measurements."
    return out
