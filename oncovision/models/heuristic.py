"""Classical image-processing fallback used when no trained weights are installed.

This is DEMO MODE. It finds regions whose intensity is a strong outlier
compared to the surrounding tissue, which lets the whole app (viewer,
measurements, 3D, chat, reports) run end to end without trained models. It is
not a medical-grade detector: it cannot tell tumor types apart, and the UI and
reports always label its output as heuristic.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold

from ..imaging.preprocess import foreground_mask, normalize
from .registry import CancerProfile

BASE_K = 3.0  # robust z-score a voxel needs to count as lesion core


def _robust_z(sl: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = sl[region]
    if values.size < 32:
        return np.zeros_like(sl)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826 + 1e-3
    return (sl - med) / mad


def _ultrasound_fov(sl: np.ndarray) -> np.ndarray:
    """The scanned fan/rectangle: everything not pure-black background, holes included."""
    fov = ndimage.binary_closing(ndimage.gaussian_filter(sl, 4) > 0.02, iterations=10)
    fov = ndimage.binary_fill_holes(fov)
    labels, n = ndimage.label(fov)
    if n == 0:
        return fov
    sizes = ndimage.sum(fov, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _search_region(sl: np.ndarray, fg: np.ndarray, profile: CancerProfile) -> np.ndarray:
    """Where lesions may appear: inside the body, away from its edge (skull, skin)."""
    if profile.modality == "Ultrasound":
        fg = _ultrasound_fov(sl)
    if not fg.any():
        return fg
    width = max(1, int(np.sqrt(fg.sum()) * 0.06))
    inner = ndimage.binary_erosion(fg, iterations=width)
    if profile.organ == "lung":
        # Nodules sit inside the dark lung fields, so search the filled lung area.
        lungs = (sl < np.percentile(sl[fg], 35)) & inner
        lungs = ndimage.binary_opening(lungs, iterations=2)
        return ndimage.binary_fill_holes(ndimage.binary_closing(lungs, iterations=4)) & inner
    return inner


def _keep_best_component(mask: np.ndarray, score: np.ndarray, min_size: int) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, labels, idx)
    strength = ndimage.mean(np.abs(score), labels, idx)
    ranking = np.where(sizes >= min_size, sizes * strength, 0)
    if ranking.max() <= 0:
        return np.zeros_like(mask)
    return labels == idx[int(np.argmax(ranking))]


def _zscores(volume: np.ndarray, profile: CancerProfile):
    """Per-slice robust z-scores inside the search region, plus region sizes."""
    sign = -1.0 if profile.lesion_polarity == "dark" else 1.0
    # Ultrasound speckle would swamp the z-scores, so smooth it away first.
    sigma = 3.0 if profile.modality == "Ultrasound" else 0.0
    zvol = np.zeros(volume.shape, dtype=np.float32)
    region_sizes = np.zeros(volume.shape[0], dtype=np.int64)
    for i, sl in enumerate(volume):
        fg = foreground_mask(sl)
        region = _search_region(sl, fg, profile)
        if region.sum() < 64:
            continue
        z = sign * _robust_z(ndimage.gaussian_filter(sl, sigma) if sigma else sl, region)
        z[~region] = 0
        zvol[i] = z
        region_sizes[i] = region.sum()
    return zvol, region_sizes


def _lesion_masks(zvol: np.ndarray, region_sizes: np.ndarray, k: float):
    """Return (lesion, core) masks for one threshold setting."""
    lesion = np.zeros(zvol.shape, dtype=bool)
    core = np.zeros(zvol.shape, dtype=bool)
    for i, z in enumerate(zvol):
        if region_sizes[i] == 0:
            continue
        cand = apply_hysteresis_threshold(z, k * 0.5, k)
        cand = ndimage.binary_opening(cand, iterations=1)
        cand = ndimage.binary_fill_holes(cand)
        min_size = max(12, int(region_sizes[i] * 0.002))
        lesion[i] = _keep_best_component(cand, z, min_size)
        core[i] = lesion[i] & (z >= k)

    if zvol.shape[0] > 1 and lesion.any():
        # Keep the largest connected 3D lesion to drop isolated slice noise.
        labels, n = ndimage.label(lesion)
        sizes = ndimage.sum(lesion, labels, range(1, n + 1))
        lesion = labels == (int(np.argmax(sizes)) + 1)
        core &= lesion
    return lesion, core


def _flair_edema(flair: np.ndarray, lesion: np.ndarray, profile: CancerProfile) -> np.ndarray:
    """FLAIR-bright tissue connected to the lesion: the usual definition of edema."""
    zf, sizes = _zscores(normalize(flair), profile)
    bright = np.zeros(zf.shape, dtype=bool)
    for i, z in enumerate(zf):
        if sizes[i]:
            bright[i] = apply_hysteresis_threshold(z, 1.0, 2.0)
    labels, _ = ndimage.label(bright | lesion)
    keep = np.unique(labels[lesion])
    return np.isin(labels, keep[keep > 0])


def _sub_regions(lesion: np.ndarray, core: np.ndarray, profile: CancerProfile, edema: np.ndarray | None) -> np.ndarray:
    labels = np.zeros(lesion.shape, dtype=np.uint8)
    if profile.num_seg_classes == 2:
        labels[lesion] = 1
        return labels
    # Brain: bright core = enhancing tumor, dark holes inside it = necrosis,
    # surrounding moderately bright (or FLAIR-bright) tissue = edema.
    if edema is not None:
        lesion = lesion | edema
    for i in range(lesion.shape[0]):
        filled = ndimage.binary_fill_holes(core[i])
        necrotic = filled & ~core[i]
        labels[i][lesion[i]] = 2
        labels[i][core[i]] = 3
        labels[i][necrotic] = 1
    return labels


def segment(volume: np.ndarray, profile: CancerProfile, flair: np.ndarray | None = None):
    """Heuristic segmentation of a (Z, H, W) volume.

    Returns ``(labels, uncertainty, saliency)``. Uncertainty comes from how
    often each voxel stays in the lesion as the threshold varies. For brain MRI
    with a FLAIR sequence, edema is taken from FLAIR as in the BraTS definition.
    """
    zvol, region_sizes = _zscores(normalize(volume), profile)
    lesion, core = _lesion_masks(zvol, region_sizes, BASE_K)
    votes = np.zeros(zvol.shape, dtype=np.float32)
    ks = (BASE_K - 0.75, BASE_K - 0.25, BASE_K + 0.25, BASE_K + 0.75)
    for k in ks:
        votes += _lesion_masks(zvol, region_sizes, k)[0]
    p = votes / len(ks)
    uncertainty = (1.0 - np.abs(2 * p - 1)).astype(np.float32)
    uncertainty[(p == 0) & ~lesion] = 0.0
    saliency = ndimage.gaussian_filter(np.clip(zvol / (2 * BASE_K), 0, 1), sigma=(0, 2, 2)).astype(np.float32)
    edema = None
    if flair is not None and flair is not volume and profile.organ == "brain" and lesion.any():
        edema = _flair_edema(flair, lesion, profile) & ~lesion
        # Edema borders are inherently fuzzy in this heuristic.
        rim = edema & ~ndimage.binary_erosion(edema, iterations=2)
        uncertainty[rim] = np.maximum(uncertainty[rim], 0.5)
    return _sub_regions(lesion, core, profile, edema), uncertainty, saliency


def abnormality_score(labels: np.ndarray, saliency: np.ndarray) -> float:
    """A 0-1 score for "something abnormal is visible", from lesion size and contrast."""
    lesion = labels > 0
    if not lesion.any():
        return float(np.clip(saliency.max() * 0.3, 0, 0.3))
    contrast = float(saliency[lesion].mean())
    size = min(1.0, lesion.sum() / 400.0)
    return float(np.clip(0.4 + 0.35 * contrast + 0.25 * size, 0, 0.99))
