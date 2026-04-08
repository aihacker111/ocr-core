"""
BaseLayoutDetector — abstract contract every layout model must satisfy.

To add a new layout model:
    1. Subclass BaseLayoutDetector.
    2. Implement `_load()` and `_predict()`.
    3. Register in layout/detectors/__init__.py.

The pipeline references only this interface, never a concrete class directly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

from ..config.layout_config import LayoutConfig
from .region import LayoutRegion

logger = logging.getLogger(__name__)


class BaseLayoutDetector(ABC):
    """
    Abstract layout detector.

    Lifecycle:
        detector = SomeDetector(config)   # __init__ calls load()
        regions  = detector.detect(image) # public entry point
    """

    def __init__(self, config: LayoutConfig) -> None:
        self.config = config
        self._ready: bool = False
        logger.info("[%s] Loading model …", self.__class__.__name__)
        self._load()
        self._ready = True
        logger.info("[%s] Ready.", self.__class__.__name__)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> list[LayoutRegion]:
        """
        Run layout detection on a PIL image.

        Returns regions sorted in natural reading order (top→bottom, left→right).
        Regions labelled ABANDON are included; callers decide whether to filter.
        """
        if not self._ready:
            raise RuntimeError(f"{self.__class__.__name__} is not loaded.")
        return self._predict(image)

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self.__class__.__name__

    # ── Subclass contract ──────────────────────────────────────────────────────

    @abstractmethod
    def _load(self) -> None:
        """
        Download / deserialize / move model to device.
        Called once in __init__. Must set self._ready = True on success.
        """

    @abstractmethod
    def _predict(self, image: Image.Image) -> list[LayoutRegion]:
        """
        Run the model on a single PIL image (already RGB).
        Must return a sorted, index-stamped list of LayoutRegion.
        """