"""
PageLoader — converts document sources into lists of PIL Images.

Supported inputs:
    • Single image file  (.png .jpg .jpeg .tiff .tif .bmp .webp)
    • PDF file           (rasterised via PyMuPDF / fitz)
    • Directory          (all supported images, sorted by filename)

Page selection (PDF and directory only):
    pages=None          — all pages (default)
    pages=1             — only page 1
    pages=[1, 3, 5]     — pages 1, 3 and 5
    pages="2-5"         — pages 2 through 5 inclusive
    pages="1,3,5-8"     — pages 1, 3, 5, 6, 7, 8
    pages=(2, 5)        — pages 2 through 5 (tuple treated as range)

All page numbers are 1-based (human-friendly).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Union

from PIL import Image

logger = logging.getLogger(__name__)

PagesSpec = Union[None, int, str, list[int], tuple[int, int]]


class PageLoader:

    SUPPORTED_IMAGE_EXTS: frozenset[str] = frozenset({
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    })

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    def load(
        self,
        source: str | Path,
        pages:  PagesSpec = None,
    ) -> list[Image.Image]:
        return list(self.iter_pages(source, pages=pages))

    def iter_pages(
        self,
        source: str | Path,
        pages:  PagesSpec = None,
    ) -> Iterator[Image.Image]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        suffix = path.suffix.lower()
        if path.is_dir():
            yield from self._load_directory(path, pages=pages)
        elif suffix == ".pdf":
            yield from self._load_pdf(path, pages=pages)
        elif suffix in self.SUPPORTED_IMAGE_EXTS:
            if pages is not None:
                logger.debug("[PageLoader] pages= ignored for single image file")
            yield self._load_image(path)
        else:
            raise ValueError(
                f"Unsupported file type: '{path.suffix}'. "
                f"Supported: .pdf, {', '.join(sorted(self.SUPPORTED_IMAGE_EXTS))}"
            )

    def page_count(self, source: str | Path) -> int:
        """Return total page count without loading any images."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if path.suffix.lower() == ".pdf":
            try:
                import fitz
            except ImportError as exc:
                raise ImportError("pip install pymupdf") from exc
            doc = fitz.open(str(path))
            n   = len(doc)
            doc.close()
            return n
        if path.is_dir():
            return sum(
                1 for f in path.iterdir()
                if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
            )
        return 1

    def _load_image(self, path: Path) -> Image.Image:
        logger.debug("[PageLoader] Loading image: %s", path.name)
        return Image.open(path).convert("RGB")

    def _load_pdf(self, path: Path, pages: PagesSpec = None) -> Iterator[Image.Image]:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF loading.\n"
                "Install with:  pip install pymupdf"
            ) from exc

        doc = fitz.open(str(path))
        n   = len(doc)
        logger.info("[PageLoader] PDF '%s' has %d page(s), dpi=%d", path.name, n, self.dpi)

        indices = self._resolve_indices(pages, total=n, source_name=path.name)
        logger.info("[PageLoader] Loading %d page(s): %s", len(indices), _format_indices(indices))

        try:
            mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            for zero_idx in indices:
                page = doc[zero_idx]
                pix  = page.get_pixmap(matrix=mat, alpha=False)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                logger.debug("[PageLoader] Page %d → %dx%d px", zero_idx + 1, pix.width, pix.height)
                yield img
        finally:
            doc.close()

    def _load_directory(self, directory: Path, pages: PagesSpec = None) -> Iterator[Image.Image]:
        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        )
        if not files:
            raise ValueError(f"No supported image files found in: {directory}")

        n       = len(files)
        indices = self._resolve_indices(pages, total=n, source_name=str(directory))
        logger.info("[PageLoader] Directory: %d image(s) selected from %d total", len(indices), n)
        for zero_idx in indices:
            yield self._load_image(files[zero_idx])

    @staticmethod
    def _resolve_indices(pages: PagesSpec, total: int, source_name: str = "") -> list[int]:
        """Convert PagesSpec (1-based) → sorted list of 0-based indices."""
        if total == 0:
            return []
        if pages is None:
            return list(range(total))

        one_based = PageLoader._parse_pages(pages)

        invalid = [p for p in one_based if p < 1 or p > total]
        if invalid:
            raise ValueError(
                f"Page(s) {invalid} out of range for '{source_name}' "
                f"(document has {total} page(s), 1-based)."
            )

        seen:    set[int]  = set()
        indices: list[int] = []
        for p in one_based:
            if p not in seen:
                seen.add(p)
                indices.append(p - 1)
        return indices

    @staticmethod
    def _parse_pages(pages: PagesSpec) -> list[int]:
        """Parse any PagesSpec into a list of 1-based page numbers."""
        if isinstance(pages, int):
            return [pages]

        if isinstance(pages, (list, tuple)):
            if (
                isinstance(pages, tuple)
                and len(pages) == 2
                and all(isinstance(p, int) for p in pages)
            ):
                start, end = pages
                if start > end:
                    raise ValueError(f"Invalid range: ({start}, {end}) — start must be ≤ end.")
                return list(range(start, end + 1))
            result: list[int] = []
            for item in pages:
                result.extend(PageLoader._parse_pages(item))
            return result

        if isinstance(pages, str):
            return PageLoader._parse_page_string(pages)

        raise TypeError(
            f"pages= must be None, int, list, tuple, or str — got {type(pages).__name__}"
        )

    @staticmethod
    def _parse_page_string(spec: str) -> list[int]:
        """Parse "1,3,5-8" → [1, 3, 5, 6, 7, 8]."""
        result: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                bounds = part.split("-", 1)
                try:
                    start = int(bounds[0].strip())
                    end   = int(bounds[1].strip())
                except ValueError:
                    raise ValueError(f"Invalid page range segment: '{part}'")
                if start > end:
                    raise ValueError(f"Invalid range '{part}': start must be ≤ end.")
                result.extend(range(start, end + 1))
            else:
                try:
                    result.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid page number: '{part}'")
        return result


def _format_indices(indices: list[int]) -> str:
    one_based = [i + 1 for i in indices]
    if len(one_based) <= 10:
        return str(one_based)
    return f"[{one_based[0]}…{one_based[-1]}] ({len(one_based)} pages)"