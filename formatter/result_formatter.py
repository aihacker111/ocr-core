"""
ResultFormatter — thin dispatch layer that converts OCRResult lists to string output.

Supported formats
-----------------
    "markdown"  Pretty-printed Markdown document (default)
    "json"      JSON array, one object per region
    "text"      Plain text — just the extracted strings, separated by blank lines
"""
from __future__ import annotations

import json

from ..ocr.result        import OCRResult
from .markdown_builder   import MarkdownBuilder


class ResultFormatter:
    """
    Usage:
        formatter = ResultFormatter()
        output    = formatter.format(results, fmt="markdown")
    """

    def __init__(self) -> None:
        self._md_builder = MarkdownBuilder()

    def format(self, results: list[OCRResult], fmt: str = "markdown") -> str:
        """
        Convert OCRResult list to string in the requested format.

        Args:
            results:  Sorted list of OCRResult (sorted by region_index).
            fmt:      "markdown" | "json" | "text"

        Returns:
            Formatted string.
        """
        fmt = fmt.lower().strip()
        if fmt == "markdown":
            return self._to_markdown(results)
        if fmt == "json":
            return self._to_json(results)
        if fmt == "text":
            return self._to_text(results)
        raise ValueError(
            f"Unknown output format '{fmt}'. "
            f"Choose from: 'markdown', 'json', 'text'."
        )

    # ── Format implementations ─────────────────────────────────────────────────

    def _to_markdown(self, results: list[OCRResult]) -> str:
        return self._md_builder.build(results)

    def _to_json(self, results: list[OCRResult]) -> str:
        data = [r.to_dict() for r in results]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _to_text(self, results: list[OCRResult]) -> str:
        parts = [
            r.text.strip()
            for r in results
            if r.text and r.text.strip()
        ]
        return "\n\n".join(parts)
