"""
Image utility helpers shared across the pipeline.
"""
from __future__ import annotations

import io
import base64

from PIL import Image


def crop_region(image: Image.Image, bbox: list[int]) -> Image.Image:
    """
    Crop a PIL image to the given [x1, y1, x2, y2] bounding box.
    Coordinates are clamped to image bounds to avoid negative slices.
    """
    w, h = image.size
    x1 = max(0, bbox[0])
    y1 = max(0, bbox[1])
    x2 = min(w, bbox[2])
    y2 = min(h, bbox[3])
    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox — return a 1×1 blank crop rather than crashing
        return Image.new("RGB", (1, 1), (255, 255, 255))
    return image.crop((x1, y1, x2, y2))


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_max(image: Image.Image, max_side: int = 2048) -> Image.Image:
    """
    Proportionally resize so the longest side does not exceed max_side.
    Returns the original image unchanged if it already fits.
    """
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB if not already."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image
