"""
TextCleaner — post-processes raw GLM-OCR output strings.

Patterns handled (fully general — no hardcoded numbers):
    $ ^{2} $   →  ²        (inline-math wrapped superscript)
    $ ^{12} $  →  ¹²       (multi-digit)
    $ _{2} $   →  ₂        (subscript)
    ^ {1}      →  ¹        (bare caret, with or without braces/spaces)
    ^1         →  ¹        (compact caret)
    _{3}       →  ₃        (bare underscore)
    \\mathrm{A c c o r d i n g}  →  According
    s c h o o l              →  school   (spaced characters)
    1 According to …         →  ¹According to …  (FOOTNOTE label only)
"""

from __future__ import annotations
import re
from ..layout.labels import DocLabel

_FORMULA_LABELS  = frozenset({DocLabel.FORMULA})
_FOOTNOTE_LABELS = frozenset({DocLabel.FOOTNOTE, DocLabel.VISION_FOOTNOTE})

_SUPERSCRIPT = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUBSCRIPT   = str.maketrans("0123456789+-=()",   "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


class TextCleaner:

    def clean(self, text: str, label: DocLabel) -> str:
        if not text or not text.strip():
            return ""
        if label in _FORMULA_LABELS:
            return text.strip()
        return self._clean_text(text, label)

    def _clean_text(self, text: str, label: DocLabel) -> str:
        text = self._unwrap_inline_math(text)
        text = self._unwrap_latex_text_commands(text)
        text = self._fix_spaced_chars(text)
        text = self._clean_caret_superscripts(text)
        text = self._clean_underscore_subscripts(text)
        if label in _FOOTNOTE_LABELS:
            text = self._fix_leading_footnote_number(text)
        return text.strip()

    def _unwrap_inline_math(self, text: str) -> str:
        """$ ^{N} $  →  ᴺ  (any digit or letter, any format)"""
        sup = re.compile(r'\$\s*\^\s*\{?([0-9a-zA-Z+\-=()]+)\}?\s*\$')
        sub = re.compile(r'\$\s*_\s*\{?([0-9a-zA-Z+\-=()]+)\}?\s*\$')
        text = sup.sub(lambda m: m.group(1).translate(_SUPERSCRIPT), text)
        text = sub.sub(lambda m: m.group(1).translate(_SUBSCRIPT),   text)
        return text

    def _unwrap_latex_text_commands(self, text: str) -> str:
        """\\mathrm{A c c}  →  A c c"""
        pattern = re.compile(
            r'\\(?:mathrm|text|textbf|textit|textrm|mbox|hbox)\s*\{([^{}]*)\}',
            re.UNICODE,
        )
        prev = None
        while prev != text:
            prev = text
            text = pattern.sub(lambda m: m.group(1), text)
        return text

    def _fix_spaced_chars(self, text: str) -> str:
        """s c h o o l  →  school  (3+ consecutive single chars)"""
        spaced = re.compile(r'(?<!\w)((?:\S\s){3,}\S)(?!\w)')
        return spaced.sub(lambda m: m.group(0).replace(' ', ''), text)

    def _clean_caret_superscripts(self, text: str) -> str:
        """^ {1}  ^{12}  ^ n  →  ¹  ¹²  ⁿ  (any remaining bare ^ not in $)"""
        pattern = re.compile(r'\^\s*\{?([0-9a-zA-Z+\-=()]+)\}?')
        return pattern.sub(lambda m: m.group(1).translate(_SUPERSCRIPT), text)

    def _clean_underscore_subscripts(self, text: str) -> str:
        """_{2}  →  ₂  (any remaining bare _ not in $)"""
        pattern = re.compile(r'_\s*\{?([0-9a-zA-Z+\-=()]+)\}?')
        return pattern.sub(lambda m: m.group(1).translate(_SUBSCRIPT), text)

    def _fix_leading_footnote_number(self, text: str) -> str:
        """
        Catch-all for FOOTNOTE / VISION_FOOTNOTE regions where the model
        outputs a bare digit marker with no LaTeX wrapping at all.
        "1 According to …"  →  "¹According to …"
        "12 See also …"     →  "¹²See also …"
        General for any number — not hardcoded.
        """
        return re.sub(
            r'^(\d+)\s+',
            lambda m: m.group(1).translate(_SUPERSCRIPT),
            text,
        )