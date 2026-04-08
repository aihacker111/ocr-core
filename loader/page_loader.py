"""
PageLoader — converts document sources into lists of PIL Images.

Supported inputs:
    • Single image file  (.png .jpg .jpeg .tiff .tif .bmp .webp)
    • PDF file           (rasterised via PyMuPDF / fitz)
    • Directory          (all supported images, sorted by filename)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

logger = logging.getLogger(__name__)


class PageLoader:
    """
    Lazy page iterator and eager list loader.

    Args:
        dpi:    Rasterisation resolution for PDF pages (default 150 dpi).
                Higher values give sharper images but slower load times.
                150 dpi is a good balance for A4/Letter documents.
    """

    SUPPORTED_IMAGE_EXTS: frozenset[str] = frozenset({
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    })

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, source: str | Path) -> list[Image.Image]:
        """
        Load all pages eagerly. For large PDFs prefer iter_pages() to avoid
        holding every page in memory simultaneously.
        """
        return list(self.iter_pages(source))

    def iter_pages(self, source: str | Path) -> Iterator[Image.Image]:
        """Yield PIL Images (RGB) one page at a time."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        suffix = path.suffix.lower()
        if path.is_dir():
            yield from self._load_directory(path)
        elif suffix == ".pdf":
            yield from self._load_pdf(path)
        elif suffix in self.SUPPORTED_IMAGE_EXTS:
            yield self._load_image(path)
        else:
            raise ValueError(
                f"Unsupported file type: '{path.suffix}'. "
                f"Supported: .pdf, {', '.join(sorted(self.SUPPORTED_IMAGE_EXTS))}"
            )

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> Image.Image:
        logger.debug("[PageLoader] Loading image: %s", path.name)
        return Image.open(path).convert("RGB")

    def _load_pdf(self, path: Path) -> Iterator[Image.Image]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF loading.\n"
                "Install with:  pip install pymupdf"
            ) from exc

        logger.info("[PageLoader] Opening PDF: %s (dpi=%d)", path.name, self.dpi)
        doc = fitz.open(str(path))
        n   = len(doc)
        try:
            mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            for page_num in range(n):
                page = doc[page_num]
                pix  = page.get_pixmap(matrix=mat, alpha=False)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                logger.debug(
                    "[PageLoader] PDF page %d/%d → %dx%d px",
                    page_num + 1, n, pix.width, pix.height,
                )
                yield img
        finally:
            doc.close()

    def _load_directory(self, directory: Path) -> Iterator[Image.Image]:
        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        )
        if not files:
            raise ValueError(
                f"No supported image files found in directory: {directory}"
            )
        logger.info("[PageLoader] Found %d image(s) in: %s", len(files), directory)
        for img_path in files:
            yield self._load_image(img_path)
