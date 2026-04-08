"""
PP-DocLayout-V3 detector — PyTorch backend via HuggingFace transformers.

Model source: PaddlePaddle/PP-DocLayoutV3_safetensors (HuggingFace)
Architecture: RT-DETR, loaded via AutoModelForObjectDetection

Weights are downloaded once and cached by the transformers library.
No ultralytics / ONNX runtime required — pure transformers inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

from ...config.layout_config import LayoutConfig
from ..base import BaseLayoutDetector
from ..labels import label_from_index
from ..postprocessor import LayoutPostprocessor
from ..region import LayoutRegion

logger = logging.getLogger(__name__)

_HF_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"


class PPDocLayoutDetector(BaseLayoutDetector):
    """
    Concrete layout detector backed by PP-DocLayout-V3 via transformers.

    Uses AutoModelForObjectDetection + AutoImageProcessor so the full
    pre/post-processing is handled by the official HuggingFace pipeline.
    """

    def __init__(self, config: LayoutConfig) -> None:
        self._model           = None
        self._image_processor = None
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
                "transformers is required for PPDocLayoutDetector.\n"
                "Install with:  pip install transformers"
            ) from exc

        model_id  = self.config.model_path or _HF_REPO
        cache_dir = str(Path(self.config.cache_dir).expanduser()) if not self.config.model_path else None
        device    = self._resolve_device()

        logger.info("[PPDocLayout] Loading image processor from: %s", model_id)
        self._image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )

        logger.info("[PPDocLayout] Loading model from: %s", model_id)
        self._model = AutoModelForObjectDetection.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        self._model.to(device)
        self._model.eval()
        logger.info("[PPDocLayout] Model loaded on device: %s", device)

    def _predict(self, image: Image.Image) -> list[LayoutRegion]:
        image = image.convert("RGB")
        raw_regions = self._run_inference(image)
        return self._post.process(raw_regions)

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_inference(self, image: Image.Image) -> list[LayoutRegion]:
        """
        Run transformers object-detection inference and convert results
        to a flat list of LayoutRegion.
        """
        device = self._resolve_device()

        # Preprocess — image_processor handles resize, normalise, tensor conversion
        inputs = self._image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Post-process — returns boxes in original image pixel space
        results = self._image_processor.post_process_object_detection(
            outputs,
            target_sizes=[image.size[::-1]],   # (height, width)
        )

        regions: list[LayoutRegion] = []
        if not results:
            return regions

        result = results[0]   # batch size = 1

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

            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
            if x2 <= x1 or y2 <= y1:
                continue

            # Convert polygon tensor → list[list[int]] if present
            poly_list = None
            if poly is not None:
                poly_list = [[int(round(p[0])), int(round(p[1]))]
                             for p in poly.tolist()]

            regions.append(LayoutRegion(
                index=idx,
                label=label_from_index(label_id.item()),
                score=score_val,
                bbox=[x1, y1, x2, y2],
                poly=poly_list,
            ))

        return regions

    # ── Device helper ──────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device