"""
LayoutConfig — configuration for the layout detection stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LayoutConfig:
    """
    All knobs for the layout detection stage.

    Attributes:
        detector_name       Key into DETECTOR_REGISTRY (e.g. "pp_doclay", "dummy").
        model_path          Optional explicit path to local weights file.
                            If set, skips HuggingFace download.
        cache_dir           Directory for cached model weights.
        input_size          (width, height) to resize pages before detection.
        score_threshold     Minimum detection confidence to keep a region.
        nms_iou_threshold   IoU threshold for Non-Maximum Suppression.
        containment_overlap Overlap ratio to suppress nested duplicate regions.
        keep_discard_regions Whether to keep ABANDON-labelled regions.
        device              "auto" | "cpu" | "cuda" | "cuda:0" etc.
    """

    detector_name:        str             = "pp_doclay"
    model_path:           Optional[str]   = None
    cache_dir:            str             = "~/.cache/ocr-core/layout"
    input_size:           Tuple[int, int] = field(default=(1024, 1024))
    score_threshold:      float           = 0.30
    nms_iou_threshold:    float           = 0.45
    containment_overlap:  float           = 0.80
    keep_discard_regions: bool            = False
    device:               str             = "auto"
