"""
PP-DocLayout-V3 detector — PyTorch backend via HuggingFace transformers.

Label mapping is read directly from model.config.id2label at load time,
so it always stays in sync with whatever the model ships — no hardcoded list.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ...config.layout_config import LayoutConfig
from ..base import BaseLayoutDetector
from ..postprocessor import LayoutPostprocessor
from ..region import LayoutRegion
from ..labels import DocLabel

logger = logging.getLogger(__name__)

_HF_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"


class PPDocLayoutDetector(BaseLayoutDetector):

    def __init__(self, config: LayoutConfig) -> None:
        self._model           = None
        self._image_processor = None
        self._id2label:  dict[int, str]      = {}   # filled in _load()
        self._post = LayoutPostprocessor(
            score_threshold=config.score_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            containment_overlap=config.containment_overlap,
            keep_discard=config.keep_discard_regions,
        )
        super().__init__(config)

    # ── BaseLayoutDetector contract ────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "transformers is required.\n"
                "Install with:  pip install transformers"
            ) from exc

        model_id  = self.config.model_path or _HF_REPO
        cache_dir = str(Path(self.config.cache_dir).expanduser()) if not self.config.model_path else None
        device    = self._resolve_device()

        self._image_processor = AutoImageProcessor.from_pretrained(
            model_id, cache_dir=cache_dir,
        )
        self._model = AutoModelForObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir,
        )
        self._model.to(device)
        self._model.eval()

        # ── Read label mapping directly from model config ──────────────────
        # Mirrors the reference example: model.config.id2label[label]
        # Keys are strings in HF config ("0", "1", …) — convert to int.
        self._id2label = {
            int(k): v
            for k, v in self._model.config.id2label.items()
        }
        logger.info(
            "[PPDocLayout] Loaded %d labels from model.config.id2label: %s",
            len(self._id2label),
            self._id2label,
        )

    def _predict(self, image: Image.Image) -> list[LayoutRegion]:
        image = image.convert("RGB")
        return self._post.process(self._run_inference(image))

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_inference(self, image: Image.Image) -> list[LayoutRegion]:
        import torch
        device = self._resolve_device()

        inputs = self._image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._image_processor.post_process_object_detection(
            outputs,
            target_sizes=[image.size[::-1]],   # (height, width)
        )

        regions: list[LayoutRegion] = []
        if not results:
            return regions

        result         = results[0]
        scores         = result["scores"]
        label_ids      = result["labels"]
        boxes          = result["boxes"]
        polygon_points = result.get("polygon_points", [None] * len(scores))

        for idx, (score, label_id, box, poly) in enumerate(
            zip(scores, label_ids, boxes, polygon_points)
        ):
            score_val = score.item()
            if score_val < self.config.score_threshold:
                continue

            label_int  = label_id.item()
            # Use model.config.id2label exactly like the reference example
            label_str  = self._id2label.get(label_int, "abandon")
            doc_label  = self._str_to_doclabel(label_str)

            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
            if x2 <= x1 or y2 <= y1:
                continue

            poly_list = None
            if poly is not None:
                poly_list = [[int(round(p[0])), int(round(p[1]))]
                             for p in poly.tolist()]

            regions.append(LayoutRegion(
                index=idx,
                label=doc_label,
                score=score_val,
                bbox=[x1, y1, x2, y2],
                poly=poly_list,
            ))

        return regions

    # ── Label string → DocLabel ────────────────────────────────────────────────

    @staticmethod
    def _str_to_doclabel(label_str: str) -> DocLabel:
        """
        Map the raw string from model.config.id2label to a DocLabel.

        Falls back to ABANDON for any unknown string so the pipeline
        never crashes on new/unexpected label names.
        """
        _MAP: dict[str, DocLabel] = {
            "abstract":           DocLabel.ABSTRACT,
            "algorithm":          DocLabel.ALGORITHM,
            "aside_text":         DocLabel.ASIDE_TEXT,
            "chart":              DocLabel.CHART,
            "content":            DocLabel.CONTENT,
            "formula":            DocLabel.FORMULA,
            "doc_title":          DocLabel.DOC_TITLE,
            "figure_title":       DocLabel.FIGURE_TITLE,
            "footer":             DocLabel.FOOTER,
            "footnote":           DocLabel.FOOTNOTE,
            "formula_number":     DocLabel.FORMULA_NUMBER,
            "header":             DocLabel.HEADER,
            "image":              DocLabel.IMAGE,
            "number":             DocLabel.NUMBER,
            "paragraph_title":    DocLabel.PARAGRAPH_TITLE,
            "reference":          DocLabel.REFERENCE,
            "reference_content":  DocLabel.REFERENCE_CONTENT,
            "seal":               DocLabel.SEAL,
            "table":              DocLabel.TABLE,
            "text":               DocLabel.TEXT,
            "vision_footnote":    DocLabel.VISION_FOOTNOTE,
        }
        return _MAP.get(label_str.lower(), DocLabel.ABANDON)

    # ── Device helper ──────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.config.device