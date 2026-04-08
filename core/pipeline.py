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

    # Full run, saving layout visualisation images:
    pipeline = OCRPipeline()
    doc = pipeline.run_file("invoice.pdf", save_layout_dir="debug/")
    # → saves  debug/page_0000_layout.png, debug/page_0001_layout.png …
    print(doc.merged_text)

    # Single image:
    result = pipeline.run_image(img, save_layout_path="debug/layout.png")
    print(result.formatted)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .config.pipeline_config     import PipelineConfig
from .layout                     import build_detector, BaseLayoutDetector
from .layout.region              import LayoutRegion
from .loader.page_loader         import PageLoader
from .ocr                        import build_ocr_model, BaseOCRModel
from .ocr.result                 import OCRResult
from .ocr.worker_pool            import WorkerPool
from .formatter.result_formatter import ResultFormatter
from .utils.visualize    import LayoutVisualizer

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """
    OCR output for a single page.

    Attributes:
        page_index      0-based index within the source document.
        regions         Raw LayoutRegion list (after postprocessing).
        results         OCRResult list sorted by reading order.
        formatted       String output in the requested format.
        layout_image    Annotated PIL image (only set when visualisation requested).
    """
    page_index:    int
    regions:       list[LayoutRegion]
    results:       list[OCRResult]
    formatted:     str
    layout_image:  Optional[Image.Image] = field(default=None, repr=False)

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
        self.config       = config or PipelineConfig()
        self._detector:   Optional[BaseLayoutDetector] = None
        self._ocr:        Optional[BaseOCRModel]       = None
        self._pool:       Optional[WorkerPool]         = None
        self._loader      = PageLoader()
        self._formatter   = ResultFormatter()
        self._visualizer  = LayoutVisualizer()

    # ── Lazy model loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
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

    def run_image(
        self,
        image:             Image.Image,
        save_layout_path:  Optional[str | Path] = None,
    ) -> PageResult:
        """
        Process a single PIL Image.

        Args:
            image:             Input image.
            save_layout_path:  If given, save the annotated layout image here
                               (e.g. "debug/layout.png").

        Returns:
            PageResult — includes .layout_image (PIL) when save_layout_path is set.
        """
        self._ensure_loaded()
        return self._process_page(
            image=image,
            page_index=0,
            save_layout_path=Path(save_layout_path) if save_layout_path else None,
        )

    def run_file(
        self,
        source:           str | Path,
        save_layout_dir:  Optional[str | Path] = None,
    ) -> DocumentResult:
        """
        Process a PDF, image file, or directory of images.

        Args:
            source:           Path to file or directory.
            save_layout_dir:  If given, layout visualisation images are saved
                              here as page_0000_layout.png, page_0001_layout.png …

        Returns:
            DocumentResult — each PageResult includes .layout_image when
            save_layout_dir is set.
        """
        self._ensure_loaded()
        source     = Path(source)
        layout_dir = Path(save_layout_dir) if save_layout_dir else None
        if layout_dir:
            layout_dir.mkdir(parents=True, exist_ok=True)

        pages = self._loader.load(source)
        logger.info("[OCRPipeline] %d page(s) loaded from: %s", len(pages), source)

        page_results: list[PageResult] = []
        for idx, image in enumerate(pages):
            logger.info("[OCRPipeline] Processing page %d / %d …", idx + 1, len(pages))

            layout_path = (
                layout_dir / f"page_{idx:04d}_layout.png"
                if layout_dir else None
            )
            page_results.append(
                self._process_page(image=image, page_index=idx, save_layout_path=layout_path)
            )

        return DocumentResult(source_path=source, pages=page_results)

    def run_file_to_string(self, source: str | Path) -> str:
        """Convenience: run_file and return all pages merged into one string."""
        return self.run_file(source).merged_text

    # ── Core processing ────────────────────────────────────────────────────────

    def _process_page(
        self,
        image:             Image.Image,
        page_index:        int,
        save_layout_path:  Optional[Path] = None,
    ) -> PageResult:
        image = image.convert("RGB")

        # Stage 1 — Layout detection
        regions = self._detector.detect(image)
        logger.debug("[OCRPipeline] Page %d: %d region(s) detected", page_index, len(regions))

        # Stage 1b — Visualise layout (optional)
        layout_image: Optional[Image.Image] = None
        if save_layout_path is not None:
            layout_image = self._visualizer.draw(image, regions)
            self._visualizer.save(image, regions, save_layout_path)
            logger.info(
                "[OCRPipeline] Layout visualisation saved → %s", save_layout_path
            )

        if not regions:
            logger.warning("[OCRPipeline] Page %d: no regions detected.", page_index)
            return PageResult(
                page_index=page_index,
                regions=[],
                results=[],
                formatted="",
                layout_image=layout_image,
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
            layout_image=layout_image,
        )