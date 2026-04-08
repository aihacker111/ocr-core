from __future__ import annotations
from ..layout.labels import DocLabel

PROMPT_TEXT    = "Text Recognition:"
PROMPT_TABLE   = "Table Recognition:"
PROMPT_FORMULA = "Formula Recognition:"

_LABEL_PROMPT_MAP: dict[DocLabel, str] = {
    DocLabel.DOC_TITLE:          PROMPT_TEXT,
    DocLabel.PARAGRAPH_TITLE:    PROMPT_TEXT,
    DocLabel.TEXT:               PROMPT_TEXT,
    DocLabel.CONTENT:            PROMPT_TEXT,
    DocLabel.ABSTRACT:           PROMPT_TEXT,
    DocLabel.ASIDE_TEXT:         PROMPT_TEXT,
    DocLabel.REFERENCE:          PROMPT_TEXT,
    DocLabel.REFERENCE_CONTENT:  PROMPT_TEXT,
    DocLabel.FOOTNOTE:           PROMPT_TEXT,
    DocLabel.ALGORITHM:          PROMPT_TEXT,
    DocLabel.FIGURE_TITLE:       PROMPT_TEXT,
    DocLabel.VISION_FOOTNOTE:    PROMPT_TEXT,
    DocLabel.FORMULA_NUMBER:     PROMPT_TEXT,
    DocLabel.HEADER:             PROMPT_TEXT,
    DocLabel.FOOTER:             PROMPT_TEXT,
    DocLabel.NUMBER:             PROMPT_TEXT,
    DocLabel.SEAL:               PROMPT_TEXT,
    DocLabel.TABLE:              PROMPT_TABLE,
    DocLabel.FORMULA:            PROMPT_FORMULA,
    DocLabel.IMAGE:              PROMPT_TEXT,
    DocLabel.CHART:              PROMPT_TEXT,
    DocLabel.ABANDON:            PROMPT_TEXT,
}


class PromptBuilder:
    def get_prompt(self, label: DocLabel) -> str:
        return _LABEL_PROMPT_MAP.get(label, PROMPT_TEXT)

    def build_messages(self, crop_image, label: DocLabel) -> list[dict]:
        return [{"role": "user", "content": [
            {"type": "image", "image": crop_image},
            {"type": "text",  "text":  self.get_prompt(label)},
        ]}]