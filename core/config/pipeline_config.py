"""
PipelineConfig — top-level configuration that groups layout + OCR settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
        save_images_dir          If set, IMAGE/CHART crops are written under this path.
        markdown_image_prefix    If set, ``image_path`` / ``![](...)`` use this prefix
                                 instead of ``save_images_dir`` (e.g. a path relative
                                 to the Markdown file while crops live on disk elsewhere).
    """

    layout:                  LayoutConfig = field(default_factory=LayoutConfig)
    ocr:                     OCRConfig    = field(default_factory=OCRConfig)
    max_workers:             int          = 4
    skip_non_text:           bool         = True
    output_format:           str          = "markdown"   # "markdown" | "json" | "text"
    save_images_dir:         Optional[str] = None
    markdown_image_prefix:   Optional[str] = None
