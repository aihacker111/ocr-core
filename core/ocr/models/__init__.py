"""
OCR model registry.

Add new models here so the pipeline can resolve them by name string.
"""

from __future__ import annotations

from ...config.ocr_config import OCRConfig
from ..base import BaseOCRModel
from .dummy   import DummyOCRModel
from .glm_ocr import GLMOCRModel

MODEL_REGISTRY: dict[str, type[BaseOCRModel]] = {
    "glm_ocr": GLMOCRModel,
    "dummy":   DummyOCRModel,
}


def build_ocr_model(config: OCRConfig) -> BaseOCRModel:
    """
    Instantiate an OCR model by config.model_name.

    Example:
        config.model_name = "glm_ocr"   # default
        config.model_name = "dummy"     # no-weight testing
    """
    cls = MODEL_REGISTRY.get(config.model_name)
    if cls is None:
        available = list(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown OCR model '{config.model_name}'. "
            f"Available: {available}"
        )
    return cls(config)


__all__ = [
    "BaseOCRModel",
    "GLMOCRModel",
    "DummyOCRModel",
    "MODEL_REGISTRY",
    "build_ocr_model",
]