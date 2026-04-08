"""
WorkerPool — dispatches OCR jobs across a thread pool.

Design decisions:
    - Uses ThreadPoolExecutor (not ProcessPoolExecutor) because PyTorch's
      GPU tensors and CUDA contexts cannot be pickled for multiprocessing.
    - The BaseOCRModel._recognize() implementation is responsible for its
      own thread-safety (typically a threading.Lock around inference).
    - Failed regions are returned as OCRResult with .error set, never raised.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from ..layout.labels import DocLabel, NON_TEXT_LABELS
from ..layout.region import LayoutRegion
from ..utils.image_utils import crop_region
from .base import BaseOCRModel
from .result import OCRResult

logger = logging.getLogger(__name__)


class WorkerPool:
    """
    Manages parallel OCR execution.

    Args:
        model:       Any BaseOCRModel implementation.
        max_workers: Thread count (default 4; CPU-bound limit on GIL, but
                     I/O wait in model prep benefits from >1).
        skip_non_text: Skip regions whose label is in NON_TEXT_LABELS
                       (figures, charts, seals) to avoid wasting inference.
    """

    def __init__(
        self,
        model:         BaseOCRModel,
        max_workers:   int  = 4,
        skip_non_text: bool = True,
    ) -> None:
        self._model         = model
        self._max_workers   = max_workers
        self._skip_non_text = skip_non_text

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        full_image: Image.Image,
        regions:    list[LayoutRegion],
    ) -> list[OCRResult]:
        """
        Run OCR on all regions in parallel.
        Returns results in the same order as regions (sorted by region.index).
        """
        jobs = self._build_jobs(full_image, regions)
        if not jobs:
            return []

        results: list[OCRResult] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_index = {
                pool.submit(self._run_single, *job): job[1]   # job[1] = region_index
                for job in jobs
            }
            for future in as_completed(future_to_index):
                results.append(future.result())

        results.sort(key=lambda r: r.region_index)
        return results

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_jobs(
        self,
        full_image: Image.Image,
        regions:    list[LayoutRegion],
    ) -> list[tuple]:
        """
        Returns list of (crop, region_index, label, bbox) tuples.
        Regions whose label is in NON_TEXT_LABELS are skipped when
        skip_non_text=True.
        """
        jobs = []
        for r in regions:
            if self._skip_non_text and r.label in NON_TEXT_LABELS:
                logger.debug("Skipping non-text region %d (%s)", r.index, r.label.value)
                continue
            crop = crop_region(full_image, r.bbox)
            jobs.append((crop, r.index, r.label, r.bbox))
        return jobs

    def _run_single(
        self,
        crop:         Image.Image,
        region_index: int,
        label:        DocLabel,
        bbox:         list[int],
    ) -> OCRResult:
        logger.debug("OCR region %d (%s) …", region_index, label.value)
        result = self._model.recognize(crop, region_index, label, bbox)
        logger.debug("OCR region %d done (%d chars)", region_index, len(result.text))
        return result