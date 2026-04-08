"""
PP-DocLayout-V3 label definitions.

23-class label set covering the full range of document regions:
structural, textual, graphical, tabular, mathematical, and metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


# ── Label enum ────────────────────────────────────────────────────────────────

class DocLabel(str, Enum):
    """Every region type PP-DocLayout-V3 can detect."""
    # — Textual ----------------------------------------------------------------
    DOC_TITLE        = "doc_title"        # main document title
    PARAGRAPH_TITLE  = "paragraph_title"  # section / subsection heading
    TEXT             = "text"             # body paragraph
    ABSTRACT         = "abstract"         # abstract block
    REFERENCE        = "reference"        # bibliography / reference entry
    LIST             = "list"             # bullet / numbered list block
    CODE             = "code"             # source-code block
    # — Graphical --------------------------------------------------------------
    FIGURE           = "figure"           # image / photo / diagram
    FIGURE_CAPTION   = "figure_caption"   # caption below a figure
    FIGURE_FOOTNOTE  = "figure_footnote"  # note anchored to a figure
    CHART            = "chart"            # bar / line / pie chart
    CHART_CAPTION    = "chart_caption"    # caption below a chart
    CHART_FOOTNOTE   = "chart_footnote"   # note anchored to a chart
    # — Tabular ----------------------------------------------------------------
    TABLE            = "table"            # data table
    TABLE_CAPTION    = "table_caption"    # caption above/below a table
    TABLE_FOOTNOTE   = "table_footnote"   # note anchored to a table
    # — Mathematical -----------------------------------------------------------
    ISOLATE_FORMULA  = "isolate_formula"  # display / block formula
    FORMULA_CAPTION  = "formula_caption"  # formula label/number
    # — Page metadata ----------------------------------------------------------
    HEADER           = "header"           # running header
    FOOTER           = "footer"           # running footer
    PAGE_NUMBER      = "page_number"      # page number
    SEAL             = "seal"             # official stamp / seal
    # — Noise / discard --------------------------------------------------------
    ABANDON          = "abandon"          # decoration, watermark, noise


# ── Index ↔ label maps ───────────────────────────────────────────────────────

# Ordered to match PP-DocLayout-V3 class indices (0-based)
LABEL_LIST: list[DocLabel] = [
    DocLabel.PARAGRAPH_TITLE,  # 0
    DocLabel.CHART,            # 1
    DocLabel.CHART_CAPTION,    # 2
    DocLabel.CHART_FOOTNOTE,   # 3
    DocLabel.DOC_TITLE,        # 4
    DocLabel.FIGURE,           # 5
    DocLabel.FIGURE_CAPTION,   # 6
    DocLabel.FIGURE_FOOTNOTE,  # 7
    DocLabel.FOOTER,           # 8
    DocLabel.HEADER,           # 9
    DocLabel.ISOLATE_FORMULA,  # 10
    DocLabel.LIST,             # 11
    DocLabel.PAGE_NUMBER,      # 12
    DocLabel.REFERENCE,        # 13
    DocLabel.SEAL,             # 14
    DocLabel.TABLE,            # 15
    DocLabel.TABLE_CAPTION,    # 16
    DocLabel.TABLE_FOOTNOTE,   # 17
    DocLabel.TEXT,             # 18
    DocLabel.ABSTRACT,         # 19
    DocLabel.CODE,             # 20
    DocLabel.FORMULA_CAPTION,  # 21
    DocLabel.ABANDON,          # 22
]

INDEX_TO_LABEL: dict[int, DocLabel] = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
LABEL_TO_INDEX: dict[DocLabel, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}


def label_from_index(index: int) -> DocLabel:
    """Return the DocLabel for a model class index, falling back to ABANDON."""
    return INDEX_TO_LABEL.get(index, DocLabel.ABANDON)


# ── Semantic group sets ───────────────────────────────────────────────────────

# Labels that carry primary reading content
CONTENT_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.DOC_TITLE,
    DocLabel.PARAGRAPH_TITLE,
    DocLabel.TEXT,
    DocLabel.ABSTRACT,
    DocLabel.REFERENCE,
    DocLabel.LIST,
    DocLabel.CODE,
})

CAPTION_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.FIGURE_CAPTION,
    DocLabel.FIGURE_FOOTNOTE,
    DocLabel.CHART_CAPTION,
    DocLabel.CHART_FOOTNOTE,
    DocLabel.TABLE_CAPTION,
    DocLabel.TABLE_FOOTNOTE,
    DocLabel.FORMULA_CAPTION,
})

GRAPHICAL_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.FIGURE,
    DocLabel.CHART,
})

TABLE_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.TABLE,
})

FORMULA_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.ISOLATE_FORMULA,
})

METADATA_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.HEADER,
    DocLabel.FOOTER,
    DocLabel.PAGE_NUMBER,
    DocLabel.SEAL,
})

DISCARD_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.ABANDON,
})

# Labels where we skip OCR (no text to extract)
NON_TEXT_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.FIGURE,
    DocLabel.CHART,
    DocLabel.SEAL,
    DocLabel.ABANDON,
})