"""
Layout detector registry.

Add new detectors here so the pipeline can resolve them by name string.
"""

from __future__ import annotations

from ...config.layout_config import LayoutConfig
from ..base import BaseLayoutDetector
from .dummy     import DummyLayoutDetector
from .pp_doclay import PPDocLayoutDetector

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps string name → class.  Add your own detector here.

DETECTOR_REGISTRY: dict[str, type[BaseLayoutDetector]] = {
    "pp_doclay":  PPDocLayoutDetector,
    "dummy":      DummyLayoutDetector,
}


def build_detector(config: LayoutConfig) -> BaseLayoutDetector:
    """
    Instantiate a layout detector by the name stored in config.detector_name.

    Example:
        config.detector_name = "pp_doclay"   # default
        config.detector_name = "dummy"       # no-weight testing
        config.detector_name = "my_custom"   # after registering in DETECTOR_REGISTRY
    """
    cls = DETECTOR_REGISTRY.get(config.detector_name)
    if cls is None:
        available = list(DETECTOR_REGISTRY)
        raise ValueError(
            f"Unknown detector '{config.detector_name}'. "
            f"Available: {available}"
        )
    return cls(config)


__all__ = [
    "BaseLayoutDetector",
    "PPDocLayoutDetector",
    "DummyLayoutDetector",
    "DETECTOR_REGISTRY",
    "build_detector",
]