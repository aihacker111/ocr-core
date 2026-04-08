"""
OCRResult — the output unit of any OCR model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..layout.labels import DocLabel


@dataclass
class OCRResult:
    """
    Text extracted from one layout region.

    Attributes:
        region_index  Matches LayoutRegion.index.
        label         Semantic label of the source region.
        text          Extracted text (may be empty on failure).
        bbox          Source bounding box [x1, y1, x2, y2].
        error         Set to an error message if recognition failed.
        image_path    Relative path to a saved PNG crop for IMAGE/CHART regions
                      (set when ``PipelineConfig.save_images_dir`` is used).
    """

    region_index: int
    label:        DocLabel
    text:         str
    bbox:         list[int]
    error:        Optional[str] = None
    image_path:   Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_image(self) -> bool:
        return self.image_path is not None

    def to_dict(self) -> dict:
        d: dict = {
            "index":   self.region_index,
            "label":   self.label.value,
            "content": self.text,
            "bbox_2d": self.bbox,
        }
        if self.error:
            d["error"] = self.error
        if self.image_path is not None:
            d["image_path"] = self.image_path
        return d