"""
PP-DocLayout-V3 label definitions.

Index→label mapping is read from model.config.id2label at runtime
inside PPDocLayoutDetector._load(), so this file no longer contains
a hardcoded LABEL_LIST.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class DocLabel(str, Enum):
    DOC_TITLE          = "doc_title"
    PARAGRAPH_TITLE    = "paragraph_title"
    TEXT               = "text"
    ABSTRACT           = "abstract"
    CONTENT            = "content"
    ASIDE_TEXT         = "aside_text"
    REFERENCE          = "reference"
    REFERENCE_CONTENT  = "reference_content"
    FOOTNOTE           = "footnote"
    ALGORITHM          = "algorithm"
    IMAGE              = "image"
    FIGURE_TITLE       = "figure_title"
    CHART              = "chart"
    VISION_FOOTNOTE    = "vision_footnote"
    TABLE              = "table"
    FORMULA            = "formula"
    FORMULA_NUMBER     = "formula_number"
    HEADER             = "header"
    FOOTER             = "footer"
    NUMBER             = "number"
    SEAL               = "seal"
    ABANDON            = "abandon"


CONTENT_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.DOC_TITLE, DocLabel.PARAGRAPH_TITLE, DocLabel.TEXT,
    DocLabel.CONTENT, DocLabel.ABSTRACT, DocLabel.ASIDE_TEXT,
    DocLabel.REFERENCE, DocLabel.REFERENCE_CONTENT,
    DocLabel.FOOTNOTE, DocLabel.ALGORITHM,
})

GRAPHICAL_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.IMAGE, DocLabel.CHART,
})

CAPTION_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.FIGURE_TITLE, DocLabel.VISION_FOOTNOTE, DocLabel.FORMULA_NUMBER,
})

TABLE_LABELS:   FrozenSet[DocLabel] = frozenset({DocLabel.TABLE})
FORMULA_LABELS: FrozenSet[DocLabel] = frozenset({DocLabel.FORMULA})

METADATA_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.HEADER, DocLabel.FOOTER, DocLabel.NUMBER, DocLabel.SEAL,
})

DISCARD_LABELS: FrozenSet[DocLabel] = frozenset({DocLabel.ABANDON})

NON_TEXT_LABELS: FrozenSet[DocLabel] = frozenset({
    DocLabel.IMAGE, DocLabel.CHART, DocLabel.SEAL, DocLabel.ABANDON,
})