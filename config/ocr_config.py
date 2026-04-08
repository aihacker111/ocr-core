"""
OCRConfig — configuration for the OCR model stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRConfig:
    """
    All knobs for the OCR recognition stage.

    Attributes:
        model_name          Key into MODEL_REGISTRY (e.g. "glm_ocr", "dummy").
        model_id            HuggingFace model ID or local path override.
                            Defaults to "zai-org/GLM-OCR" when None.
        cache_dir           Directory for cached HF weights.
        device              "auto" | "cpu" | "cuda" | "cuda:0" etc.
        dtype               "auto" | "float32" | "float16" | "bfloat16".
        max_new_tokens      Maximum tokens the model may generate per region.
        temperature         Sampling temperature; 0.0 = greedy (deterministic).
        trust_remote_code   Passed to AutoModel/AutoProcessor.from_pretrained().
    """

    model_name:         str           = "glm_ocr"
    model_id:           Optional[str] = None
    cache_dir:          str           = "~/.cache/ocr-core/ocr"
    device:             str           = "auto"
    dtype:              str           = "auto"
    max_new_tokens:     int           = 2048
    temperature:        float         = 0.0
    trust_remote_code:  bool          = True
