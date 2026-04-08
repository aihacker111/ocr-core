"""
PromptBuilder — maps a DocLabel to the correct GLM-OCR prompt string.

GLM-OCR supports three recognition modes selectable by the prompt text:
    "Text Recognition:"     — general text, lists, references, code, …
    "Table Recognition:"    — structured tables (returns Markdown table)
    "Formula Recognition:"  — LaTeX mathematical expressions

Centralising this logic here means changing a prompt never requires
touching model or pipeline code.
"""

from __future__ import annotations

from ..layout.labels import DocLabel

# ── Prompt constants ──────────────────────────────────────────────────────────

PROMPT_TEXT    = "Text Recognition:"
PROMPT_TABLE   = "Table Recognition:"
PROMPT_FORMULA = "Formula Recognition:"

# ── Label → prompt mapping ────────────────────────────────────────────────────

_LABEL_PROMPT_MAP: dict[DocLabel, str] = {
    # Textual content
    DocLabel.DOC_TITLE:       PROMPT_TEXT,
    DocLabel.PARAGRAPH_TITLE: PROMPT_TEXT,
    DocLabel.TEXT:            PROMPT_TEXT,
    DocLabel.ABSTRACT:        PROMPT_TEXT,
    DocLabel.REFERENCE:       PROMPT_TEXT,
    DocLabel.LIST:            PROMPT_TEXT,
    DocLabel.CODE:            PROMPT_TEXT,
    # Captions & footnotes
    DocLabel.FIGURE_CAPTION:  PROMPT_TEXT,
    DocLabel.FIGURE_FOOTNOTE: PROMPT_TEXT,
    DocLabel.CHART_CAPTION:   PROMPT_TEXT,
    DocLabel.CHART_FOOTNOTE:  PROMPT_TEXT,
    DocLabel.TABLE_CAPTION:   PROMPT_TEXT,
    DocLabel.TABLE_FOOTNOTE:  PROMPT_TEXT,
    DocLabel.FORMULA_CAPTION: PROMPT_TEXT,
    # Page metadata
    DocLabel.HEADER:          PROMPT_TEXT,
    DocLabel.FOOTER:          PROMPT_TEXT,
    DocLabel.PAGE_NUMBER:     PROMPT_TEXT,
    DocLabel.SEAL:            PROMPT_TEXT,
    # Structural — special modes
    DocLabel.TABLE:           PROMPT_TABLE,
    DocLabel.ISOLATE_FORMULA: PROMPT_FORMULA,
    # Non-text (should be filtered before reaching here, but be safe)
    DocLabel.FIGURE:          PROMPT_TEXT,
    DocLabel.CHART:           PROMPT_TEXT,
    DocLabel.ABANDON:         PROMPT_TEXT,
}


class PromptBuilder:
    """
    Returns the correct OCR prompt for a given DocLabel.

    Can be subclassed or replaced to customise prompts for other models.
    """

    def get_prompt(self, label: DocLabel) -> str:
        """Return the model prompt for the given region label."""
        return _LABEL_PROMPT_MAP.get(label, PROMPT_TEXT)

    def build_messages(
        self,
        crop_image,          # PIL.Image
        label: DocLabel,
    ) -> list[dict]:
        """Build the HuggingFace chat-template messages list for one region."""
        prompt = self.get_prompt(label)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_image},
                    {"type": "text",  "text":  prompt},
                ],
            }
        ]