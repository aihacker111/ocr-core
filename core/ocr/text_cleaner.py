"""
TextCleaner — post-processes raw GLM-OCR output strings.
"""

from __future__ import annotations

import re

from ..layout.labels import DocLabel

_FORMULA_LABELS = frozenset({DocLabel.FORMULA})

_SUPERSCRIPT = str.maketrans("0123456789+-=()niaβ", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵝ")
_SUBSCRIPT   = str.maketrans("0123456789+-=()aeinoruvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₙₒᵣᵤᵥₓ")


class TextCleaner:
    """Stateless cleaner. Call clean(text, label) after every GLM-OCR decode."""

    def clean(self, text: str, label: DocLabel) -> str:
        if not text or not text.strip():
            return ""
        if label in _FORMULA_LABELS:
            return text.strip()
        return self._clean_text(text)

    # ── Internal pipeline ──────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        text = self._unwrap_inline_math_superscripts(text)   # ← new, runs first
        text = self._unwrap_latex_text_commands(text)
        text = self._fix_spaced_chars(text)
        text = self._clean_superscript_markers(text)
        text = self._clean_subscript_markers(text)
        return text.strip()

    def _unwrap_inline_math_superscripts(self, text: str) -> str:
        """
        Convert inline math that contains only a superscript/subscript into
        plain Unicode, then remove the $ delimiters entirely.

        Handles these patterns:
            $ ^{2} $          →  ²
            $^{2}$            →  ²
            $ ^2 $            →  ²
            $ _{2} $          →  ₂
            $ ^{12} $         →  ¹²
            $ ^{2} $$ ^{2} $  →  ²²   (repeated inline math)

        General inline math that isn't just a super/subscript is left alone.
        """
        # Pattern: optional spaces, $, optional space, ^/{digits or chars}, optional space, $
        sup_inline = re.compile(
            r'\$\s*\^\s*\{?([0-9a-zA-Z+\-=()]+)\}?\s*\$'
        )
        sub_inline = re.compile(
            r'\$\s*_\s*\{?([0-9a-zA-Z+\-=()]+)\}?\s*\$'
        )

        def _to_sup(m: re.Match) -> str:
            return m.group(1).translate(_SUPERSCRIPT)

        def _to_sub(m: re.Match) -> str:
            return m.group(1).translate(_SUBSCRIPT)

        text = sup_inline.sub(_to_sup, text)
        text = sub_inline.sub(_to_sub, text)
        return text

    def _unwrap_latex_text_commands(self, text: str) -> str:
        """
        \\mathrm{A c c o r d i n g}  →  A c c o r d i n g
        \\text{hello}                 →  hello
        """
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
        """
        "s c h o o l  s y s t e m"  →  "school system"
        Requires 3+ consecutive single-char groups to avoid false positives.
        """
        spaced = re.compile(r'(?<!\w)((?:\S\s){3,}\S)(?!\w)')
        return spaced.sub(lambda m: m.group(0).replace(' ', ''), text)

    def _clean_superscript_markers(self, text: str) -> str:
        """
        Bare LaTeX superscript notation not wrapped in $:
            ^{12}  →  ¹²
            ^ 1    →  ¹
        """
        pattern = re.compile(r'\^\s*\{?(\d+)\}?')
        return pattern.sub(lambda m: m.group(1).translate(_SUPERSCRIPT), text)

    def _clean_subscript_markers(self, text: str) -> str:
        """
        _{2}  →  ₂
        """
        pattern = re.compile(r'_\s*\{?(\d+)\}?')
        return pattern.sub(lambda m: m.group(1).translate(_SUBSCRIPT), text)