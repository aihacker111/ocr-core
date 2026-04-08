"""
LayoutRegion — the output unit of any layout detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .labels import DocLabel


@dataclass
class LayoutRegion:
    """
    A single detected region returned by a layout detector.

    Attributes:
        index       Reading-order index assigned after sorting.
        label       Semantic class of the region (DocLabel enum).
        score       Detection confidence in [0, 1].
        bbox        Bounding box as [x1, y1, x2, y2] in image pixel coords.
        poly        Optional multi-point polygon (for skewed / curved docs).
    """

    index: int
    label: DocLabel
    score: float
    bbox:  list[int]          # [x1, y1, x2, y2]
    poly:  Optional[list[list[int]]] = field(default=None, repr=False)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "index": self.index,
            "label": self.label.value,
            "score": round(self.score, 4),
            "bbox":  self.bbox,
        }
        if self.poly is not None:
            d["poly"] = self.poly
        return d