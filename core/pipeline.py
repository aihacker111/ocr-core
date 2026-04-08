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
    from core import OCRPipeline, PipelineConfig, LayoutConfig, OCRConfig

    # All pages:
    pipeline = OCRPipeline()
    doc = pipeline.run_file("report.pdf")
    print(doc.merged_text)

    # Specific pages (1-based):
    doc = pipeline.run_file("report.pdf", pages=1)          # page 1 only
    doc = pipeline.run_file("report.pdf", pages=[1, 3, 5])  # pages 1, 3, 5
    doc = pipeline.run_file("report.pdf", pages="2-5")      # pages 2 to 5
    doc = pipeline.run_file("report.pdf", pages="1,3,5-8")  # mixed
    doc = pipeline.run_file("report.pdf", pages=(2, 5))     # range tuple

    # Single image:
    result = pipeline.run_image(img, save_layout_path="debug/layout.png")
    print(result.formatted)

    # With layout visualisation:
    doc = pipeline.run_file("report.pdf", pages="1-3", save_layout_dir="debug/")
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
from .loader.page_loader         import PageLoader, PagesSpec
from .ocr                        import build_ocr_model, BaseOCRModel
from .ocr.result                 import OCRResult
from .ocr.worker_pool            import WorkerPool
from .formatter.result_formatter import ResultFormatter
from .utils.layout_visualizer    import LayoutVisualizer

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """
    OCR output for a single page.

    Attributes:
        page_index      Original 0-based page index in the source document.
        regions         LayoutRegion list after postprocessing.
        results         OCRResult list sorted by reading order.
        formatted       String output in the requested format.
        layout_image    Annotated PIL image (set when visualisation requested).
    """
    page_index:   int
    regions:      list[LayoutRegion]
    results:      list[OCRResult]
    formatted:    str
    layout_image: Optional[Image.Image] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"PageResult(page={self.page_index + 1}, "
            f"regions={len(self.regions)}, "
            f"ocr_results={len(self.results)})"
        )


@dataclass
class DocumentResult:
    """
    Aggregated result for a whole document (all processed pages).

    Attributes:
        source_path  Original file path (None if run_image was used).
        pages        One PageResult per processed page, in order.
    """
    source_path: Optional[Path]
    pages:       list[PageResult] = field(default_factory=list)

    @property
    def merged_text(self) -> str:
        """All pages merged into one string, separated by '---'."""
        return "\n\n---\n\n".join(p.formatted for p in self.pages)

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
        self.config      = config or PipelineConfig()
        self._detector:  Optional[BaseLayoutDetector] = None
        self._ocr:       Optional[BaseOCRModel]       = None
        self._pool:      Optional[WorkerPool]         = None
        self._loader     = PageLoader()
        self._formatter  = ResultFormatter()
        self._visualizer = LayoutVisualizer()

    # ── Lazy model loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load layout detector and OCR model if not already loaded."""
        if self._detector is None:
            logger.info(
                "[OCRPipeline] Initialising layout detector '%s' …",
                self.config.layout.detector_name,
            )
            self._detector = build_detector(self.config.layout)

        if self._ocr is None:
            logger.info(
                "[OCRPipeline] Initialising OCR model '%s' …",
                self.config.ocr.model_name,
            )
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
        image:            Image.Image,
        save_layout_path: Optional[str | Path] = None,
    ) -> PageResult:
        """
        Process a single PIL Image.

        Args:
            image:             Input RGB image.
            save_layout_path:  If given, save the annotated layout image here
                               (e.g. "debug/layout.png").

        Returns:
            PageResult with OCR results and formatted output.
            .layout_image (PIL) is set when save_layout_path is provided.
        """
        self._ensure_loaded()
        return self._process_page(
            image=image,
            page_index=0,
            save_layout_path=Path(save_layout_path) if save_layout_path else None,
        )

    def run_file(
        self,
        source:          str | Path,
        pages:           PagesSpec = None,
        save_layout_dir: Optional[str | Path] = None,
    ) -> DocumentResult:
        """
        Process a PDF, image file, or directory of images.

        Args:
            source:           Path to PDF, image file, or directory.
            pages:            Which pages to process (1-based). None = all pages.
                              Accepted formats:
                                  pages=1            → page 1 only
                                  pages=[1, 3, 5]    → pages 1, 3 and 5
                                  pages="2-5"        → pages 2 through 5 inclusive
                                  pages="1,3,5-8"    → pages 1, 3, 5, 6, 7, 8
                                  pages=(2, 5)       → pages 2 through 5 inclusive
            save_layout_dir:  If given, annotated layout images are saved here.
                              Filenames use the original 1-based page number:
                                  page_0001_layout.png, page_0003_layout.png …

        Returns:
            DocumentResult — one PageResult per processed page, in order.

        Raises:
            ValueError:      If pages spec is out of range or malformed.
            FileNotFoundError: If source path does not exist.
        """
        self._ensure_loaded()
        source     = Path(source)
        layout_dir = Path(save_layout_dir) if save_layout_dir else None
        if layout_dir:
            layout_dir.mkdir(parents=True, exist_ok=True)

        # Load only the requested pages as PIL images
        raw_pages = self._loader.load(source, pages=pages)

        # Resolve original 0-based indices for logging and layout filenames
        orig_indices = PageLoader._resolve_indices(
            pages,
            total=self._loader.page_count(source),
            source_name=str(source),
        )

        logger.info(
            "[OCRPipeline] Processing %d page(s) from: %s",
            len(raw_pages), source,
        )

        page_results: list[PageResult] = []
        for load_order, (image, orig_idx) in enumerate(zip(raw_pages, orig_indices)):
            logger.info(
                "[OCRPipeline] Page %d/%d (document page %d) …",
                load_order + 1, len(raw_pages), orig_idx + 1,
            )
            layout_path = (
                layout_dir / f"page_{orig_idx + 1:04d}_layout.png"
                if layout_dir else None
            )
            page_results.append(
                self._process_page(
                    image=image,
                    page_index=orig_idx,
                    save_layout_path=layout_path,
                )
            )

        return DocumentResult(source_path=source, pages=page_results)

    def run_file_to_string(
        self,
        source: str | Path,
        pages:  PagesSpec = None,
    ) -> str:
        """
        Convenience wrapper: run_file and return all pages merged into one string.

        Args:
            source: Path to PDF, image file, or directory.
            pages:  Which pages to process (1-based). None = all pages.
        """
        return self.run_file(source, pages=pages).merged_text

    # ── Core processing ────────────────────────────────────────────────────────

    def _process_page(
        self,
        image:            Image.Image,
        page_index:       int,
        save_layout_path: Optional[Path] = None,
    ) -> PageResult:
        """
        Run the full pipeline on a single RGB PIL image.

        Steps:
            1. Ensure RGB
            2. Layout detection  → sorted LayoutRegion list
            3. Layout visualise  → save annotated image (optional)
            4. Worker pool OCR   → OCRResult list (same order)
            5. Format output     → string in configured format
        """
        image = image.convert("RGB")

        # Stage 1 — Layout detection
        regions = self._detector.detect(image)
        logger.debug(
            "[OCRPipeline] Page %d: %d region(s) detected",
            page_index + 1, len(regions),
        )

        # Stage 2 — Layout visualisation (optional)
        layout_image: Optional[Image.Image] = None
        if save_layout_path is not None:
            layout_image = self._visualizer.draw(image, regions)
            self._visualizer.save(image, regions, save_layout_path)
            logger.info("[OCRPipeline] Layout saved → %s", save_layout_path)

        if not regions:
            logger.warning(
                "[OCRPipeline] Page %d: no regions detected.", page_index + 1
            )
            return PageResult(
                page_index=page_index,
                regions=[],
                results=[],
                formatted="",
                layout_image=layout_image,
            )

        # Stage 3 — OCR
        ocr_results = self._pool.run(full_image=image, regions=regions)
        logger.debug(
            "[OCRPipeline] Page %d: %d OCR result(s)",
            page_index + 1, len(ocr_results),
        )

        # Stage 4 — Formatting
        formatted = self._formatter.format(ocr_results, fmt=self.config.output_format)

        return PageResult(
            page_index=page_index,
            regions=regions,
            results=ocr_results,
            formatted=formatted,
            layout_image=layout_image,
        )