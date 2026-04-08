"""
PipelineConfig — top-level configuration that groups layout + OCR settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .layout_config import LayoutConfig
from .ocr_config    import OCRConfig


@dataclass
class PipelineConfig:
    """
    Aggregated configuration for the full OCR pipeline.

    Attributes:
        layout          Layout detection settings.
        ocr             OCR model settings.
        max_workers     Thread pool size for parallel OCR.
        skip_non_text   Skip figure / chart / seal regions (no text to extract).
        output_format   "markdown" | "json" | "text"
    """

    layout:        LayoutConfig = field(default_factory=LayoutConfig)
    ocr:           OCRConfig    = field(default_factory=OCRConfig)
    max_workers:   int          = 4
    skip_non_text: bool         = True
    output_format: str          = "markdown"   # "markdown" | "json" | "text"
