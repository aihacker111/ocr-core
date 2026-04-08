"""
Layout postprocessor.

Responsibilities:
    1. Score threshold filtering
    2. Non-maximum suppression (NMS)
    3. Containment analysis — remove small boxes fully inside a larger one
       of the same label (prevents duplicate regions)
    4. Reading-order sorting (top→bottom, left→right with column awareness)
    5. Re-indexing
"""

from __future__ import annotations

import numpy as np

from .labels import DocLabel, DISCARD_LABELS
from .region import LayoutRegion


class LayoutPostprocessor:
    """
    Stateless postprocessor; create one per pipeline config.
    """

    def __init__(
        self,
        score_threshold:     float = 0.30,
        nms_iou_threshold:   float = 0.45,
        containment_overlap: float = 0.80,   # inner/inner_area ratio to suppress
        keep_discard:        bool  = False,   # whether to keep ABANDON regions
    ) -> None:
        self.score_threshold     = score_threshold
        self.nms_iou_threshold   = nms_iou_threshold
        self.containment_overlap = containment_overlap
        self.keep_discard        = keep_discard

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        """Full postprocessing pipeline. Returns final, re-indexed regions."""
        regions = self._threshold(regions)
        regions = self._nms(regions)
        regions = self._remove_contained(regions)
        if not self.keep_discard:
            regions = [r for r in regions if r.label not in DISCARD_LABELS]
        regions = self._sort_reading_order(regions)
        for idx, r in enumerate(regions):
            r.index = idx
        return regions

    # ── Filtering steps ────────────────────────────────────────────────────────

    def _threshold(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        return [r for r in regions if r.score >= self.score_threshold]

    def _nms(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        if not regions:
            return regions
        regions = sorted(regions, key=lambda r: r.score, reverse=True)
        keep: list[LayoutRegion] = []
        for candidate in regions:
            if all(
                _iou(candidate.bbox, kept.bbox) < self.nms_iou_threshold
                for kept in keep
            ):
                keep.append(candidate)
        return keep

    def _remove_contained(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        """
        If region A is almost entirely inside region B (same label),
        discard A if A.area < B.area.
        """
        keep: list[LayoutRegion] = []
        for i, r in enumerate(regions):
            if r.label != DocLabel.ABANDON:
                dominated = False
                for j, other in enumerate(regions):
                    if i == j or r.label != other.label:
                        continue
                    overlap = _intersection_area(r.bbox, other.bbox)
                    if r.area > 0 and overlap / r.area >= self.containment_overlap:
                        if r.area < other.area:
                            dominated = True
                            break
                if dominated:
                    continue
            keep.append(r)
        return keep

    # ── Reading-order sort ─────────────────────────────────────────────────────

    def _sort_reading_order(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        """
        Two-level sort:
          1. Assign each region to a vertical band (row-band height = median region height).
          2. Within each band, sort left-to-right.

        This approximates natural multi-column reading order without a
        full column-detection pass.
        """
        if not regions:
            return regions

        heights = [r.height for r in regions]
        band_h  = float(np.median(heights)) if heights else 50.0

        def sort_key(r: LayoutRegion) -> tuple[int, int]:
            band  = int(r.bbox[1] / band_h)
            return (band, r.bbox[0])

        return sorted(regions, key=sort_key)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    inter = _intersection_area(a, b)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _intersection_area(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))