"""Turn arrays into viewable PNGs: segmentation overlays, heatmaps, uncertainty maps."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..models.registry import CancerProfile

LAYERS = ("original", "overlay", "heatmap", "uncertainty")

# Perceptually ordered "inferno"-like color ramp for heatmaps.
_HEAT_STOPS = np.array(
    [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float32
)
_UNC_STOPS = np.array([[0, 0, 0], [76, 29, 149], [168, 85, 247], [244, 114, 182], [255, 255, 255]], dtype=np.float32)


def _ramp(values: np.ndarray, stops: np.ndarray) -> np.ndarray:
    pos = np.clip(values, 0, 1) * (len(stops) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(stops) - 1)
    frac = (pos - lo)[..., None]
    return stops[lo] * (1 - frac) + stops[hi] * frac


def gray_rgb(base: np.ndarray) -> np.ndarray:
    g = (np.clip(base, 0, 1) * 255).astype(np.float32)
    return np.repeat(g[..., None], 3, axis=-1)


def overlay(base: np.ndarray, labels: np.ndarray, profile: CancerProfile, alpha: float = 0.45) -> np.ndarray:
    rgb = gray_rgb(base)
    for lbl in profile.display_labels:
        mask = labels == lbl.index
        if not mask.any():
            continue
        color = np.array(lbl.color, dtype=np.float32)
        rgb[mask] = rgb[mask] * (1 - alpha) + color * alpha
        edge = mask & ~ndimage.binary_erosion(mask)
        rgb[edge] = color
    return rgb


def blend_map(base: np.ndarray, values: np.ndarray, stops: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = gray_rgb(base)
    colored = _ramp(values, stops)
    weight = (alpha * np.clip(values, 0, 1) ** 0.7)[..., None]
    return rgb * (1 - weight) + colored * weight


def render_layer(layer: str, base, labels, heat, unc, profile: CancerProfile) -> np.ndarray:
    if layer == "overlay":
        return overlay(base, labels, profile)
    if layer == "heatmap":
        return blend_map(base, heat, _HEAT_STOPS)
    if layer == "uncertainty":
        return blend_map(base, unc, _UNC_STOPS, alpha=0.75)
    return gray_rgb(base)


def to_png(rgb: np.ndarray, max_side: int | None = None) -> bytes:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def panel(base, labels, heat, unc, profile: CancerProfile, panel_size: int = 384, layers=LAYERS[:3]) -> bytes:
    """Side-by-side captioned panels of one slice, used for the LLM and PDF report."""
    tiles = []
    for layer in layers:
        img = Image.fromarray(np.clip(render_layer(layer, base, labels, heat, unc, profile), 0, 255).astype(np.uint8))
        img = img.resize((panel_size, int(panel_size * img.height / img.width)), Image.LANCZOS)
        canvas = Image.new("RGB", (img.width, img.height + 22), (17, 24, 39))
        canvas.paste(img, (0, 22))
        ImageDraw.Draw(canvas).text((6, 5), layer.upper(), fill=(229, 231, 235))
        tiles.append(canvas)
    out = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), max(t.height for t in tiles)), (0, 0, 0))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 4
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
