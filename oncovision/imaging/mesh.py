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


def _body(vol: np.ndarray) -> np.ndarray:
    """The scanned part of the body as one smooth, closed region (largest 3D region of the smoothed scan)."""
    body = ndimage.gaussian_filter(vol, 1.5) > 0.08
    comp, n = ndimage.label(body)
    if n:
        body = comp == (np.argmax(ndimage.sum(body, comp, range(1, n + 1))) + 1)
    body = ndimage.binary_closing(np.pad(body, 3), iterations=3)[3:-3, 3:-3, 3:-3]
    return np.stack([ndimage.binary_fill_holes(s) for s in body])


def _chest_wall(body: np.ndarray) -> int | None:
    """Row of the chest wall: median over axial slices of the first row (from the front, row 0)
    where the body is nearly as wide as the torso."""
    walls = []
    for s in body:
        widths = s.sum(axis=1)
        if widths.max() > 0:
            walls.append(np.where(widths >= 0.85 * widths.max())[0][0])
    return int(np.median(walls)) if walls else None


def _head_first(vol: np.ndarray, body: np.ndarray, wall: int) -> bool:
    """Whether the head end of a whole-chest scan is at slice 0, judged from the anatomy.

    The standard (RAS) orientation puts the head end at the last slice, but some converted research
    scans (e.g. the Duke cases of MAMA-MIA) store the slices the other way round. Behind the chest
    wall on the patient's right, the foot end shows the liver (medium grey) and the head end the
    lung (dark). Only a clear difference (1.4x) counts; otherwise the stored orientation is kept.
    """
    level = []
    for z in range(body.shape[0]):
        s = body[z]
        if not s.any():
            level.append(np.nan)
            continue
        cols = np.where(s.any(axis=0))[0]
        region = s.copy()
        region[: wall + 5] = False
        region[:, (cols[0] + cols[-1]) // 2:] = False  # column 0 is the patient's right
        region = ndimage.binary_erosion(region, iterations=3)
        level.append(np.median(vol[z][region]) if region.sum() > 50 else np.nan)
    level = np.array(level)
    k = max(2, len(level) // 5)
    if np.isnan(level[:k]).all() or np.isnan(level[-k:]).all():
        return False
    first, last = np.nanmean(level[:k]), np.nanmean(level[-k:])
    return bool(last > 1.4 * first)


def breast_head_first(volume: np.ndarray, spacing) -> bool:
    """_head_first for a stored breast MRI volume (False for scans of one breast only)."""
    if not spacing or volume.shape[0] < 8 or volume.shape[2] * spacing[2] < 280:
        return False
    factor = max(1, int(np.ceil(max(volume.shape) / MAX_DIM)))
    vol = normalize(volume[::factor, ::factor, ::factor])
    body = _body(vol)
    wall = _chest_wall(body)
    return wall is not None and _head_first(vol, body, wall)


# Signed distance functions (mm) for the illustrated body: negative inside.
def _sd_ellipsoid(p, center, radii, n=2.0):
    q = np.abs((p - np.asarray(center)) / np.asarray(radii))
    return ((q ** n).sum(-1) ** (1 / n) - 1) * min(radii)


def _sd_capsule(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _torso_sdf(p: np.ndarray, k: float) -> np.ndarray:
    """A female upper body (chest, abdomen, shoulders, neck, arms) in mm, scaled by k.

    p is (..., 3) as (up, anterior, lateral) with the origin at the centre of the chest wall at
    breast level. Proportions are those of an average adult woman; it is an illustration for
    orientation, not anatomy from the scan.
    """
    d = _sd_ellipsoid(p, (30 * k, -100 * k, 0), (200 * k, 100 * k, 160 * k), 2.5)  # rib cage
    d = _smin(d, _sd_ellipsoid(p, (-190 * k, -95 * k, 0), (170 * k, 90 * k, 135 * k)), 60 * k)  # abdomen
    d = _smin(d, _sd_capsule(p, (185 * k, -100 * k, -150 * k), (185 * k, -100 * k, 150 * k), 55 * k), 40 * k)  # shoulders
    d = _smin(d, _sd_capsule(p, (180 * k, -100 * k, 0), (330 * k, -90 * k, 0), 48 * k), 30 * k)  # neck
    for side in (-1, 1):  # arms hanging at the sides
        d = _smin(d, _sd_capsule(p, (165 * k, -100 * k, side * 180 * k), (-340 * k, -95 * k, side * 250 * k), 44 * k), 6 * k)
    return d


def _breast_figure(vol: np.ndarray, body: np.ndarray, lab: np.ndarray, sp) -> dict | None:
    """Breast MRI showing the whole chest: the breasts, their glandular tissue and an illustrated
    upper body around them, all in the scan's coordinates (mm).

    The chest wall is one row for the whole scan (see _chest_wall). How far the breast tissue
    stands out in front of it is a height map over the chest (slice x column); smoothing that map
    gives the breasts' natural rounded surface without the ragged edges of fat and glands on MRI.
    The body is placed with its chest wall on the scan's, its width matched to the patient's chest.
    """
    if body.shape[2] * sp[2] < 280:  # one breast only (see measure.ONE_BREAST_FOV_MM)
        return None
    wall = _chest_wall(body)
    if wall is None:
        return None
    n, h, w = body.shape
    # Height of the tissue in front of the wall (mm), counted from the wall forward without gaps.
    height = body[:, :wall][:, ::-1].cumprod(axis=1).sum(axis=1) * sp[1]
    height = ndimage.gaussian_filter(height, (6 / sp[0], 6 / sp[2]))  # ~6 mm smoothing
    # Round off the breasts where the scan stops above and below them (instead of a flat cut).
    to_edge = np.minimum(np.arange(n), n - 1 - np.arange(n)) * sp[0]
    t = np.clip(to_edge / 30.0, 0, 1)[:, None]
    height *= np.sqrt(1 - (1 - t) ** 2)
    if height.max() < 10:
        return None

    up = -1.0 if _head_first(vol, body, wall) else 1.0  # +1: the head end is at the last slice
    chest_cols = np.where(body[:, wall].any(axis=0))[0]
    if chest_cols.size < 2:
        return None
    centre_col = (chest_cols[0] + chest_cols[-1]) / 2
    k = float(np.clip((chest_cols[-1] - chest_cols[0]) * sp[2] / 320, 0.8, 1.25))
    level = int(np.argmax(height.max(axis=1)))  # slice where the breasts stand out most
    origin = np.array([level * sp[0], wall * sp[1], centre_col * sp[2]])  # (slice, row, col) mm

    def to_body(points):  # (slice, row, col) mm -> (up, anterior, lateral) mm
        return np.stack([up * (points[..., 0] - origin[0]), origin[1] - points[..., 1], points[..., 2] - origin[2]], -1)

    zz, rr, cc = np.meshgrid(np.arange(n) * sp[0], np.arange(h) * sp[1], np.arange(w) * sp[2], indexing="ij")
    torso = _torso_sdf(to_body(np.stack([zz, rr, cc], -1)), k)

    # The breasts: where the tissue clearly stands out (the chest's flanks in front of the wall
    # line are left to the illustrated body), reaching back into the body.
    depth = (wall - np.arange(h))[None, :, None] * sp[1]  # mm in front of the wall
    domes = (depth <= height[:, None, :]) & (depth >= -60) & (height[:, None, :] > max(8.0, 0.2 * height.max()))
    sd_breast = (ndimage.distance_transform_edt(~domes, sampling=sp) - ndimage.distance_transform_edt(domes, sampling=sp))
    blend = 30 * k  # mm: how softly the breasts join the chest
    breasts = (_smin(torso, sd_breast, blend) < 0) & (torso > -3)  # in front of the body, a little overlap
    if breasts.sum() < 1000:
        return None

    # Body = illustrated torso joined smoothly with the patient's breasts.
    step = 7.0 * k
    ys, as_, xs = (np.arange(lo, hi, step) for lo, hi in ((-400 * k, 390 * k), (-230 * k, 120 * k), (-310 * k, 310 * k)))
    grid = np.stack(np.meshgrid(ys, as_, xs, indexing="ij"), -1)
    scan = np.stack([origin[0] + up * grid[..., 0], origin[1] - grid[..., 1], origin[2] + grid[..., 2]], -1)  # mm
    idx = [scan[..., i] / sp[i] for i in range(3)]
    sd_b = ndimage.map_coordinates(sd_breast.astype(np.float32), idx, order=1, mode="constant", cval=1e3)
    sdf = _smin(_torso_sdf(grid, k), sd_b, blend)
    figure = None
    try:
        verts, faces, _, _ = marching_cubes(-sdf, level=0.0, spacing=(step, step, step))
        verts += np.array([ys[0], as_[0], xs[0]])
        verts = np.stack([origin[0] + up * verts[:, 0], origin[1] - verts[:, 1], origin[2] + verts[:, 2]], -1)
        figure = {"vertices": np.round(verts, 1).ravel().tolist(), "faces": faces.astype(int).ravel().tolist()}
    except (ValueError, RuntimeError):
        pass

    # Glandular tissue: the brighter (fat-suppressed MRI) or darker (no fat suppression) class of
    # breast tissue, away from the skin and outside the tumor. Approximate, from brightness alone.
    from skimage.filters import threshold_otsu

    inner = ndimage.binary_erosion(domes & breasts & body, iterations=2) & ~ndimage.binary_dilation(lab > 0, iterations=1)
    glands = None
    if inner.sum() > 500:
        smooth = ndimage.gaussian_filter(vol, 0.7)
        bright = smooth > threshold_otsu(smooth[inner])
        glands = inner & (bright if bright[inner].mean() < 0.5 else ~bright)
        glands &= height[:, None, :] > 0.4 * height.max()  # within the breasts, not the chest between them
        comp, count = ndimage.label(glands)
        if count:
            sizes = ndimage.sum(glands, comp, range(1, count + 1))
            glands = np.isin(comp, np.where(sizes >= 6)[0] + 1)
    return {"breasts": breasts, "figure": figure, "glands": glands, "flip": up < 0}


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
    elif profile.organ == "breast":
        body = _body(vol)
        figure = _breast_figure(vol, body, lab, sp)
        if figure:
            anatomy, out["anatomy_label"] = figure["breasts"], "Breasts"
            out["flip_vertical"] = figure["flip"]  # the viewer turns the scene so the head is up
        else:
            anatomy, out["anatomy_label"] = body, "Breast (scanned area)"
    else:
        anatomy = ndimage.binary_fill_holes(np.stack([foreground_mask(s) for s in vol]))
        out["anatomy_label"] = "Brain / head" if profile.organ == "brain" else "Body outline"
    surface = _surface(anatomy, sp, step=2, sigma=2.5 if profile.organ == "breast" else 1.6)
    if surface:
        out["meshes"].append({"name": "anatomy", "color": [200, 200, 210], "opacity": 0.12, **surface})
    if profile.organ == "breast" and figure:
        if figure["figure"]:
            out["meshes"].append({"name": "body", "description": "Body (illustration for orientation, not from the scan)",
                                  "color": [96, 165, 250], "opacity": 0.1, **figure["figure"]})
        surface = _surface(figure["glands"], sp, sigma=0.8) if figure["glands"] is not None else None
        if surface:
            out["meshes"].append({"name": "glands", "description": "Glandular tissue (approx., from image brightness)",
                                  "color": [230, 170, 60], "opacity": 0.4, **surface})
    for lbl in profile.display_labels:
        mask = lab == lbl.index
        surface = _surface(mask, sp)
        if surface:
            out["meshes"].append({"name": lbl.name, "description": lbl.description, "color": list(lbl.color), "opacity": 0.9, **surface})
            if profile.organ == "breast" and figure:  # one glow per tumor spot in the illustrated view
                comp, count = ndimage.label(mask)
                spots = []
                for i, (sl, size) in enumerate(zip(ndimage.find_objects(comp), ndimage.sum(mask, comp, range(1, count + 1))), 1):
                    centre = ndimage.center_of_mass(comp[sl] == i)
                    mm = [(c + s_.start) * d for c, s_, d in zip(centre, sl, sp)]
                    spots.append([round(v, 1) for v in mm] + [round(float((3 * size * np.prod(sp) / (4 * np.pi)) ** (1 / 3)), 1)])
                out["meshes"][-1]["spots"] = spots  # (slice, row, col, radius) in mm
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
