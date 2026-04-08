"""
DummyOCRModel — returns a placeholder string for every region.

Useful for pipeline integration tests without needing GPU or model weights.
"""

from __future__ import annotations

from PIL import Image

from ...config.ocr_config import OCRConfig
from ...layout.labels import DocLabel
from ..base import BaseOCRModel


class DummyOCRModel(BaseOCRModel):
    """Returns '[DUMMY: <label>]' for every crop."""

    def _load(self) -> None:
        pass  # nothing to load

    def _recognize(self, crop: Image.Image, label: DocLabel) -> str:
        return f"[DUMMY OCR: {label.value}]"