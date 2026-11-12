"""Pixel-content heuristics for detecting blank or near-uniform image regions.

Used by both the layout predictor (drop hallucinated layout blocks over empty
space) and the recognition predictor (drop hallucinated text blocks from
full-page OCR, decide whether an empty full-page output is a correct blank-page
read or a failure).

Two signals, combined:
  * near-white fraction — most pixels have every RGB channel above a threshold
  * pixel-value standard deviation — the region is essentially one color
    (catches uniform-color fills that the white check misses)
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# Per-channel value at/above which a pixel is considered "near-white".
# Tolerates the small noise typical of PDF renders at 96 DPI.
BLANK_WHITE_THRESHOLD = 245
# Fraction of pixels that must be near-white for a region to count as blank.
BLANK_PIXEL_FRACTION = 0.99
# Pixel-value std below which a region is "essentially one color" regardless
# of what that color is (catches solid-fill rectangles, dark banners, etc.).
UNIFORM_COLOR_STD = 8.0


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return np.asarray(rgb)


def near_white_fraction(
    image: Image.Image, white_threshold: int = BLANK_WHITE_THRESHOLD
) -> float:
    """Fraction of pixels where every RGB channel ≥ ``white_threshold``."""
    arr = _as_rgb_array(image)
    if arr.size == 0:
        return 0.0
    return float(np.all(arr >= white_threshold, axis=-1).mean())


def is_blank_region(
    image: Image.Image,
    *,
    white_threshold: int = BLANK_WHITE_THRESHOLD,
    blank_pixel_fraction: float = BLANK_PIXEL_FRACTION,
    uniform_color_std: float = UNIFORM_COLOR_STD,
) -> bool:
    """True iff the image is essentially blank — either mostly near-white or
    near-uniform color. Use this on a per-block crop or a whole page.

    Returns False for empty (0-pixel) crops so callers don't accidentally
    treat a degenerate bbox as blank.
    """
    arr = _as_rgb_array(image)
    if arr.size == 0:
        return False
    if np.all(arr >= white_threshold, axis=-1).mean() > blank_pixel_fraction:
        return True
    # Per-channel std — a uniform solid color (e.g., red banner with RGB=(200,50,50))
    # has each channel constant across pixels, but mixing channels inflates the
    # aggregate std. Check each channel independently.
    per_channel_std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
    if float(per_channel_std.max()) < uniform_color_std:
        return True
    return False
