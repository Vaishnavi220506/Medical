"""3D surface meshes (marching cubes) for the browser 3D viewer."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from ..models.registry import CancerProfile
from .lung import lung_mask
from .measure import _diameters
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

    z, h, w = volume.shape
    out: dict = {
        "spacing": sp, "voxel_mm": list(spacing), "shape": list(volume.shape), "meshes": [],
        "organ": profile.organ, "modality": profile.modality, "units": "mm",
        # Axial slice plane in viewer coordinates (x = column, y = slice, z = -row).
        "slice_plane": {"center_xz": [(w - 1) / 2 * spacing[2], -(h - 1) / 2 * spacing[1]], "size": [w * spacing[2], h * spacing[1]]},
    }
    lungs = lung_mask(vol) if profile.organ == "lung" else None
    if lungs is not None and lungs.any():  # chest CT: show the lungs, not the outer body shell
        anatomy, out["anatomy_label"] = lungs, "Lungs"
    else:
        anatomy = ndimage.binary_fill_holes(np.stack([foreground_mask(s) for s in vol]))
        out["anatomy_label"] = "Brain / head" if profile.organ == "brain" else "Body outline"
    surface = _surface(anatomy, sp, step=2, sigma=1.6)
    if surface:
        out["meshes"].append({"name": "anatomy", "color": [200, 200, 210], "opacity": 0.12, **surface})
    for lbl in profile.display_labels:
        surface = _surface(lab == lbl.index, sp)
        if surface:
            out["meshes"].append({"name": lbl.name, "description": lbl.description, "color": list(lbl.color), "opacity": 0.9, **surface})
    return out


PSEUDO_SIZE = 128  # working resolution for the estimated 3D preview

# Depth of real lesions relative to their width seen on one slice: extent along the viewing
# axis divided by sqrt(longest x shortest diameter) of the largest cross-section, pooled over
# axial, coronal and sagittal views. (median, 90th percentile), measured on BraTS 2023 gliomas
# (450), MAMA-MIA breast cancers (480) and LIDC-IDRI lung nodules (303).
DEPTH_RATIO = {"brain": (1.05, 1.54), "breast": (1.07, 1.60), "lung": (0.96, 1.33)}


def _inflate(mask2d: np.ndarray, depth: int, ratio: float | None = None, flatten: float = 1.0) -> np.ndarray:
    """Turn a 2D region into a rounded 3D shape (Z, H, W), centred in depth.

    Each pixel gets a half-thickness of H * sqrt(1 - (1 - d/R)^2), where d is its distance to the
    region's edge and R the largest such distance in that region (an ellipsoid for an ellipse).
    H is R by default (a sphere for a disk); with ``ratio`` the total depth is ``ratio`` times
    sqrt(longest x shortest diameter) of the region, as in real lesions. Each connected region
    is handled separately, so a small spot stays small next to a large one.
    """
    d = ndimage.distance_transform_edt(mask2d)
    comp, n = ndimage.label(mask2d)
    r = np.ones(n + 1)
    h = np.zeros(n + 1)
    for i in range(1, n + 1):
        region = comp == i
        r[i] = max(d[region].max(), 1.0)
        if ratio is None:
            h[i] = r[i]
        else:
            longest, short = _diameters(region, 1.0, 1.0)
            h[i] = ratio * np.sqrt(longest * short) / 2
    t = np.clip(d / r[comp], 0, 1)
    half = h[comp] * np.sqrt(np.clip(1 - (1 - t) ** 2, 0, None)) * flatten
    half[~mask2d] = 0
    z = np.abs(np.arange(depth) - (depth - 1) / 2)
    return z[:, None, None] <= half[None]


def build_pseudo_meshes(volume: np.ndarray, labels: np.ndarray, profile: CancerProfile, spacing=None) -> dict:
    """Estimated 3D preview from ONE 2D image, for visualization only.

    The depth of a lesion is unknown from a single slice. Its most likely shape uses the median
    depth-to-width ratio of real 3D lesions of this organ (DEPTH_RATIO); a faint envelope shows how
    deep it could plausibly be (90th percentile). Nothing here is a measurement: the API marks the
    result as pseudo and the viewer shows no volumes.
    """
    if volume.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Pseudo-3D preview expects one image slice")
    img, lab = normalize(volume[0]), labels[0]
    factor = max(1, int(np.ceil(max(img.shape) / PSEUDO_SIZE)))
    img, lab = img[::factor, ::factor], lab[::factor, ::factor]
    pixel = (spacing[1] if spacing else 1.0) * factor
    sp = (pixel, pixel, pixel)
    median, high = DEPTH_RATIO.get(profile.organ, (1.0, 1.4))

    body = ndimage.binary_fill_holes(foreground_mask(img))
    r_body = ndimage.distance_transform_edt(body).max() if body.any() else 1.0
    lesion = lab > 0
    lesion_depth = 0.0
    if lesion.any():
        longest, short = _diameters(lesion, 1.0, 1.0)
        lesion_depth = high * np.sqrt(longest * short)
    depth = int(max(2 * r_body * 0.8, lesion_depth)) + 6
    h, w = img.shape
    out: dict = {
        "spacing": sp, "voxel_mm": list(sp), "shape": [h, depth, w], "meshes": [], "pseudo": True,
        "organ": profile.organ, "modality": profile.modality, "units": "mm" if spacing else "px",
        "depth_ratio": {"median": median, "p90": high},
        "image_factor": factor,  # mesh grid = uploaded image downsampled by this factor
    }

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
    if lesion.sum() >= 4:
        surface = _surface(upright(_inflate(lesion, depth, ratio=high)), sp, sigma=1.0)
        if surface:
            out["meshes"].append({"name": "possible_extent", "description": "Possible extent (deeper than 90% of real tumors)",
                                  "color": [148, 163, 184], "opacity": 0.12, **surface})
    for mask, value in levels:
        if mask.sum() < 4:
            continue
        lbl = profile.label(value)
        surface = _surface(upright(_inflate(mask, depth, ratio=median)), sp, sigma=1.0)
        if surface:
            out["meshes"].append({"name": lbl.name, "description": lbl.description, "color": list(lbl.color), "opacity": 0.9, **surface})
    out["warning"] = (
        "Depth estimated from one image: the solid shape is the most likely depth (median of real "
        f"{profile.organ} tumors), the faint shell how deep it could be. Not a measurement."
    )
    return out
