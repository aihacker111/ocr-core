"""
Layout preprocessor — converts a PIL image into a model-ready tensor batch.

Handles letterbox resizing to preserve aspect ratio, with configurable
padding colour and optional channel-wise normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessResult:
    """Holds the preprocessed tensor and the metadata needed to invert it."""
    tensor:   np.ndarray          # shape (1, 3, H, W), float32
    scale:    float               # resize scale applied (same for both axes)
    pad_left: int                 # horizontal padding added
    pad_top:  int                 # vertical padding added
    orig_w:   int
    orig_h:   int


class LayoutPreprocessor:
    """
    Letterbox-resizes an image to a fixed target size, then builds a
    normalised float32 CHW tensor ready for any YOLO / RT-DETR detector.
    """

    # ImageNet statistics used by PP-DocLayout-V3
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _PAD_VALUE = 114  # grey pad matching YOLO convention

    def __init__(
        self,
        input_size: tuple[int, int] = (1024, 1024),
        normalize: bool = True,
    ) -> None:
        self.input_size = input_size   # (width, height)
        self.normalize  = normalize

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, image: Image.Image) -> PreprocessResult:
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        resized, scale, pad_left, pad_top = self._letterbox(image)
        tensor = self._to_tensor(resized)

        return PreprocessResult(
            tensor=tensor,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=orig_w,
            orig_h=orig_h,
        )

    def invert_bbox(
        self,
        box: list[float] | np.ndarray,
        meta: PreprocessResult,
    ) -> list[int]:
        """
        Map a [x1, y1, x2, y2] box from letterbox-space back to original image pixels.
        """
        x1 = int(np.clip((box[0] - meta.pad_left) / meta.scale, 0, meta.orig_w))
        y1 = int(np.clip((box[1] - meta.pad_top)  / meta.scale, 0, meta.orig_h))
        x2 = int(np.clip((box[2] - meta.pad_left) / meta.scale, 0, meta.orig_w))
        y2 = int(np.clip((box[3] - meta.pad_top)  / meta.scale, 0, meta.orig_h))
        return [x1, y1, x2, y2]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _letterbox(
        self, image: Image.Image
    ) -> tuple[Image.Image, float, int, int]:
        target_w, target_h = self.input_size
        orig_w, orig_h = image.size

        scale  = min(target_w / orig_w, target_h / orig_h)
        new_w  = int(orig_w * scale)
        new_h  = int(orig_h * scale)
        pad_left = (target_w - new_w) // 2
        pad_top  = (target_h - new_h) // 2

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas  = Image.new("RGB", (target_w, target_h), (self._PAD_VALUE,) * 3)
        canvas.paste(resized, (pad_left, pad_top))

        return canvas, scale, pad_left, pad_top

    def _to_tensor(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image, dtype=np.float32) / 255.0
        if self.normalize:
            arr = (arr - self._MEAN) / self._STD
        arr = arr.transpose(2, 0, 1)          # HWC → CHW
        return arr[np.newaxis]                 # → NCHW batch of 1