"""
OCRPipeline — end-to-end document OCR orchestrator.

Pipeline stages
---------------
1. PageLoader       — reads a PDF / image file into PIL Images (one per page)
2. LayoutDetector   — detects and classifies regions on each page
3. WorkerPool+Model — runs OCR on each text region in parallel threads
4. ResultFormatter  — converts OCRResult objects to Markdown / JSON / text

Quick start
-----------
    from ocr_core.pipeline import OCRPipeline
    from ocr_core.config   import PipelineConfig, LayoutConfig, OCRConfig

    # Use dummy models (no weights needed — for testing):
    cfg = PipelineConfig(
        layout=LayoutConfig(detector_name="dummy"),
        ocr=OCRConfig(model_name="dummy"),
    )

    pipeline = OCRPipeline(cfg)
    result   = pipeline.run_image(some_pil_image)
    print(result.formatted)

    # Full GLM-OCR + PP-DocLayout on a PDF:
    pipeline = OCRPipeline()   # uses defaults
    pages    = pipeline.run_file("invoice.pdf")
    for page in pages:
        print(f"--- page {page.page_index + 1} ---")
        print(page.formatted)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .config.pipeline_config import PipelineConfig
from .layout                 import build_detector, BaseLayoutDetector
from .layout.region          import LayoutRegion
from .loader.page_loader     import PageLoader
from .ocr                    import build_ocr_model, BaseOCRModel
from .ocr.result             import OCRResult
from .ocr.worker_pool        import WorkerPool
from .formatter.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """
    OCR output for a single page.

    Attributes:
        page_index  0-based index within the source document.
        regions     Raw LayoutRegion list (after postprocessing).
        results     OCRResult list sorted by reading order.
        formatted   String output in the requested format.
    """
    page_index: int
    regions:    list[LayoutRegion]
    results:    list[OCRResult]
    formatted:  str

    def __repr__(self) -> str:
        return (
            f"PageResult(page={self.page_index}, "
            f"regions={len(self.regions)}, "
            f"ocr_results={len(self.results)})"
        )


@dataclass
class DocumentResult:
    """
    Aggregated result for a whole document (all pages).

    Attributes:
        source_path  Original file path (or None if run_image was used).
        pages        One PageResult per page, in order.
    """
    source_path: Optional[Path]
    pages:       list[PageResult] = field(default_factory=list)

    @property
    def merged_text(self) -> str:
        """All pages merged into one string (pages separated by '---')."""
        sep = "\n\n---\n\n"
        return sep.join(p.formatted for p in self.pages)

    def __repr__(self) -> str:
        return f"DocumentResult(pages={len(self.pages)}, source='{self.source_path}')"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class OCRPipeline:
    """
    Full OCR pipeline: load → detect layout → OCR → format.

    Models are loaded lazily on the first call to run_image() / run_file(),
    so instantiation is cheap.

    Args:
        config: PipelineConfig (uses sensible defaults if None).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config     = config or PipelineConfig()
        self._detector: Optional[BaseLayoutDetector] = None
        self._ocr:      Optional[BaseOCRModel]       = None
        self._pool:     Optional[WorkerPool]         = None
        self._loader    = PageLoader()
        self._formatter = ResultFormatter()

    # ── Lazy model loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load layout detector and OCR model if not already loaded."""
        if self._detector is None:
            logger.info("[OCRPipeline] Initialising layout detector '%s' …",
                        self.config.layout.detector_name)
            self._detector = build_detector(self.config.layout)

        if self._ocr is None:
            logger.info("[OCRPipeline] Initialising OCR model '%s' …",
                        self.config.ocr.model_name)
            self._ocr = build_ocr_model(self.config.ocr)

        if self._pool is None:
            self._pool = WorkerPool(
                model=self._ocr,
                max_workers=self.config.max_workers,
                skip_non_text=self.config.skip_non_text,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_image(self, image: Image.Image) -> PageResult:
        """
        Process a single PIL Image.

        Returns:
            PageResult with OCR results and formatted output.
        """
        self._ensure_loaded()
        return self._process_page(image=image, page_index=0)

    def run_file(self, source: str | Path) -> DocumentResult:
        """
        Process a PDF, image file, or directory of images.

        Returns:
            DocumentResult containing one PageResult per page.
        """
        self._ensure_loaded()
        source = Path(source)
        pages  = self._loader.load(source)
        logger.info("[OCRPipeline] %d page(s) loaded from: %s", len(pages), source)

        page_results: list[PageResult] = []
        for idx, image in enumerate(pages):
            logger.info("[OCRPipeline] Processing page %d / %d …", idx + 1, len(pages))
            page_results.append(self._process_page(image=image, page_index=idx))

        return DocumentResult(source_path=source, pages=page_results)

    def run_file_to_string(self, source: str | Path) -> str:
        """
        Convenience wrapper: run_file and return all pages merged into one string.
        """
        return self.run_file(source).merged_text

    # ── Core processing ────────────────────────────────────────────────────────

    def _process_page(self, image: Image.Image, page_index: int) -> PageResult:
        """
        Run the full pipeline on a single RGB PIL image.

        Steps:
            1. Ensure RGB
            2. Layout detection → sorted LayoutRegion list
            3. Worker pool OCR  → OCRResult list (same order)
            4. Format output string
        """
        image = image.convert("RGB")

        # Stage 1 — Layout detection
        regions = self._detector.detect(image)
        logger.debug("[OCRPipeline] Page %d: %d region(s) detected", page_index, len(regions))

        if not regions:
            logger.warning("[OCRPipeline] Page %d: no regions detected.", page_index)
            return PageResult(
                page_index=page_index,
                regions=[],
                results=[],
                formatted="",
            )

        # Stage 2 — OCR
        ocr_results = self._pool.run(full_image=image, regions=regions)
        logger.debug(
            "[OCRPipeline] Page %d: %d OCR result(s)", page_index, len(ocr_results)
        )

        # Stage 3 — Formatting
        formatted = self._formatter.format(ocr_results, fmt=self.config.output_format)

        return PageResult(
            page_index=page_index,
            regions=regions,
            results=ocr_results,
            formatted=formatted,
        )
