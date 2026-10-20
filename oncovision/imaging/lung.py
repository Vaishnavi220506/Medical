"""Find the lungs in a chest CT, so nodule search can be limited to them.

Works on a volume already clipped to the lung window (-1000..400 HU) and scaled
to [0, 1], as both the app and the 3D lung model's training data are.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

AIR = 0.43  # -400 HU in the lung window: lung tissue is darker, soft tissue brighter


def lung_mask(vol: np.ndarray) -> np.ndarray:
    """(Z, H, W) bool mask of both lungs, with nodules and vessels inside them filled in."""
    air = vol < AIR
    comp, n = ndimage.label(air)
    if n == 0:
        return np.zeros_like(air)
    # Air outside the body touches the image sides; the lungs don't.
    edge = np.unique(np.concatenate([comp[:, 0].ravel(), comp[:, -1].ravel(), comp[:, :, 0].ravel(), comp[:, :, -1].ravel()]))
    sizes = ndimage.sum(air, comp, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if i + 1 not in edge and s >= 0.002 * air.size]
    if not keep:
        return np.zeros_like(air)
    lungs = np.isin(comp, keep)
    # Close and fill holes slice by slice (nodules and vessels are bright spots inside the dark
    # lung). Padding first keeps the closing from eroding away the lungs at the volume's edges,
    # which would empty the mask of a single slice and trim the first and last slices of a CT.
    return np.stack([ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(s, 2), iterations=2)[2:-2, 2:-2]) for s in lungs])


def lung_box(vol: np.ndarray, margin: int = 10) -> tuple[slice, slice, slice] | None:
    """Bounding box of the lungs plus a margin, or None if no lungs are found."""
    lungs = lung_mask(vol)
    if not lungs.any():
        return None
    idx = [np.where(lungs.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    return tuple(slice(max(i[0] - margin, 0), min(i[-1] + margin + 1, s)) for i, s in zip(idx, vol.shape))
