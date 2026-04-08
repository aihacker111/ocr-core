"""
BaseOCRModel — abstract contract every OCR model must satisfy.

To add a new OCR model:
    1. Subclass BaseOCRModel.
    2. Implement `_load()` and `_recognize()`.
    3. Register in ocr/models/__init__.py.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from PIL import Image

from ..config.ocr_config import OCRConfig
from ..layout.labels import DocLabel
from .result import OCRResult

logger = logging.getLogger(__name__)


class BaseOCRModel(ABC):
    """
    Abstract OCR model.

    Subclasses must implement:
        _load()       — download / load weights
        _recognize()  — run inference on a single cropped PIL image
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._ready: bool = False
        logger.info("[%s] Loading model …", self.__class__.__name__)
        self._load()
        self._ready = True
        logger.info("[%s] Ready.", self.__class__.__name__)

    # ── Public API ─────────────────────────────────────────────────────────────

    def recognize(
        self,
        crop:         Image.Image,
        region_index: int,
        label:        DocLabel,
        bbox:         list[int],
    ) -> OCRResult:
        """
        Recognise text in a cropped image region.

        Wraps _recognize() with uniform error handling so the worker pool
        never propagates exceptions — failures become OCRResult.error instead.
        """
        if not self._ready:
            raise RuntimeError(f"{self.__class__.__name__} is not loaded.")
        try:
            text = self._recognize(crop, label)
            return OCRResult(region_index=region_index, label=label, text=text, bbox=bbox)
        except Exception as exc:
            logger.warning(
                "[%s] Region %d recognition failed: %s",
                self.__class__.__name__, region_index, exc,
            )
            return OCRResult(
                region_index=region_index, label=label,
                text="", bbox=bbox, error=str(exc),
            )

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    # ── Subclass contract ──────────────────────────────────────────────────────

    @abstractmethod
    def _load(self) -> None:
        """Download / deserialize / move model to device."""

    @abstractmethod
    def _recognize(self, crop: Image.Image, label: DocLabel) -> str:
        """
        Run OCR on a single RGB crop.
        Returns the extracted text string.
        Must be thread-safe (the worker pool may call this concurrently).
        """