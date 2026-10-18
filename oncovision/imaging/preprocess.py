"""Intensity normalization, resizing and foreground (body/head) masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def normalize(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Robustly rescale intensities to [0, 1] using percentiles of non-zero voxels."""
    arr = arr.astype(np.float32)
    values = arr[arr > arr.min()]
    if values.size < 16:
        values = arr.ravel()
    lo, hi = np.percentile(values, [low, high])
    if hi - lo < 1e-6:  # e.g. a flat region on a zero background
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def resize2d(img: np.ndarray, size: tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize a 2D (or channel-first 3D) array to ``size`` (H, W)."""
    if img.shape[-2:] == tuple(size):
        return img
    factors = [1.0] * (img.ndim - 2) + [size[0] / img.shape[-2], size[1] / img.shape[-1]]
    out = ndimage.zoom(img, factors, order=order)
    return out[..., : size[0], : size[1]]


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Mask of the imaged body part (head, breast, chest) in a normalized 2D slice."""
    if img.max() - img.min() < 1e-6:
        return np.zeros(img.shape, dtype=bool)
    smooth = ndimage.gaussian_filter(img, 1.0)
    try:
        thr = threshold_otsu(smooth)
    except ValueError:
        return np.zeros(img.shape, dtype=bool)
    mask = smooth > thr * 0.5
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def estimate_brain_spacing(volume: np.ndarray) -> float | None:
    """Estimate in-plane mm/pixel for a brain image of unknown scale.

    Uses the widest head cross-section and an average adult head width of
    ~145 mm. Only used for plain images, and always flagged as an estimate.
    """
    norm = normalize(volume)
    widths = []
    for sl in norm:
        mask = foreground_mask(sl)
        if mask.sum() < 0.05 * mask.size:
            continue
        cols = np.where(mask.any(axis=0))[0]
        widths.append(cols[-1] - cols[0] + 1)
    if not widths:
        return None
    return 145.0 / float(max(widths))
