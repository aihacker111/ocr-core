"""
DummyLayoutDetector — returns a single full-page text region.

Useful for unit tests, CI pipelines, or running the OCR core in isolation
without needing PP-DocLayout-V3 weights.
"""

from __future__ import annotations

from PIL import Image

from ...config.layout_config import LayoutConfig
from ..base import BaseLayoutDetector
from ..labels import DocLabel
from ..region import LayoutRegion


class DummyLayoutDetector(BaseLayoutDetector):
    """Zero-dependency detector that treats the whole image as one text block."""

    def _load(self) -> None:
        pass  # nothing to load

    def _predict(self, image: Image.Image) -> list[LayoutRegion]:
        w, h = image.size
        return [
            LayoutRegion(
                index=0,
                label=DocLabel.TEXT,
                score=1.0,
                bbox=[0, 0, w, h],
            )
        ]