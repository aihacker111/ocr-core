"""
MarkdownBuilder — converts OCRResult objects into a Markdown document.

Each region label maps to a different Markdown formatting convention:
    DOC_TITLE        → H1 heading
    PARAGRAPH_TITLE  → H2 heading
    CODE             → fenced code block
    ISOLATE_FORMULA  → display math block ($$…$$)
    TABLE            → raw Markdown table (GLM-OCR already returns Markdown)
    FIGURE/CHART captions → italicised caption
    HEADER/FOOTER/PAGE_NUMBER → HTML comment (keeps metadata without cluttering output)
    Everything else  → plain paragraph
"""
from __future__ import annotations

from ..layout.labels import DocLabel
from ..ocr.result    import OCRResult


class MarkdownBuilder:
    """
    Stateless builder: call build() with a list of OCRResult sorted by
    reading order (region_index).
    """

    def build(self, results: list[OCRResult]) -> str:
        """
        Convert OCR results into a single Markdown string.

        Regions with empty text are silently skipped.
        Results should be pre-sorted by region_index (pipeline guarantees this).
        """
        parts: list[str] = []
        for result in results:
            text = result.text.strip() if result.text else ""
            if not text:
                continue
            parts.append(self._format_block(result.label, text))

        return "\n\n".join(parts)

    # ── Formatting rules ───────────────────────────────────────────────────────

    def _format_block(self, label: DocLabel, text: str) -> str:
        if label == DocLabel.DOC_TITLE:
            return f"# {text}"

        if label == DocLabel.PARAGRAPH_TITLE:
            return f"## {text}"

        if label == DocLabel.ABSTRACT:
            return f"> {text}"

        if label == DocLabel.CODE:
            return f"```\n{text}\n```"

        if label == DocLabel.ISOLATE_FORMULA:
            # GLM-OCR returns LaTeX; wrap in display-math fences
            return f"$$\n{text}\n$$"

        if label == DocLabel.TABLE:
            # GLM-OCR "Table Recognition" returns a Markdown table directly
            return text

        if label in (DocLabel.FIGURE_CAPTION, DocLabel.CHART_CAPTION,
                     DocLabel.TABLE_CAPTION, DocLabel.FORMULA_CAPTION):
            return f"*{text}*"

        if label in (DocLabel.FIGURE_FOOTNOTE, DocLabel.CHART_FOOTNOTE,
                     DocLabel.TABLE_FOOTNOTE):
            return f"_{text}_"

        if label in (DocLabel.HEADER, DocLabel.FOOTER, DocLabel.PAGE_NUMBER):
            # Preserve metadata without disrupting reading flow
            return f"<!-- {label.value}: {text} -->"

        # Default: plain paragraph (TEXT, LIST, REFERENCE, SEAL, …)
        return text
