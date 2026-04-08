from __future__ import annotations
import re
from ..layout.labels import DocLabel
from ..ocr.result    import OCRResult

_FORMULA_SIGNALS = re.compile(
    r'[=\+\-\*/\\^_{}]|\\[a-zA-Z]+|[∑∫∂αβγδεζηθλμπσφψω]'
)


class MarkdownBuilder:

    def build(self, results: list[OCRResult]) -> str:
        parts = []
        for result in results:
            if result.label in (DocLabel.IMAGE, DocLabel.CHART):
                parts.append(self._format_graphical(result))
                continue
            text = result.text.strip() if result.text else ""
            if not text:
                continue
            parts.append(self._format_block(result.label, text))
        return "\n\n".join(parts)

    def _format_graphical(self, result: OCRResult) -> str:
        if result.image_path:
            alt = "chart" if result.label == DocLabel.CHART else "image"
            return f"![{alt}]({result.image_path})"
        kind = "chart" if result.label == DocLabel.CHART else "image"
        return f"<!-- {kind}: region {result.region_index} (no crop saved) -->"

    def _format_block(self, label: DocLabel, text: str) -> str:
        if label == DocLabel.DOC_TITLE:        return f"# {text}"
        if label == DocLabel.PARAGRAPH_TITLE:  return f"## {text}"
        if label == DocLabel.ABSTRACT:         return f"> {text}"
        if label == DocLabel.ALGORITHM:        return f"```\n{text}\n```"
        if label == DocLabel.FORMULA:
            return f"$$\n{text}\n$$" if _FORMULA_SIGNALS.search(text) else text
        if label == DocLabel.TABLE:            return text
        if label in (DocLabel.FIGURE_TITLE, DocLabel.VISION_FOOTNOTE, DocLabel.FORMULA_NUMBER):
            return f"*{text}*"
        if label in (DocLabel.HEADER, DocLabel.FOOTER, DocLabel.NUMBER):
            return f"<!-- {label.value}: {text} -->"
        return text