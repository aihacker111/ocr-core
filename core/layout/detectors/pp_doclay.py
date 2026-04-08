"""
PP-DocLayout-V3 detector — PyTorch backend via ultralytics.

Model source: PaddlePaddle/PP-DocLayoutV3_safetensors (HuggingFace)
Architecture: RT-DETR / YOLO variant, loaded via ultralytics.YOLO / RTDETR

The safetensors weights are downloaded once and cached locally.
No ONNX runtime required — pure PyTorch inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ...config.layout_config import LayoutConfig
from ...utils.downloader import ModelDownloader
from ..base import BaseLayoutDetector
from ..labels import label_from_index
from ..postprocessor import LayoutPostprocessor
from ..preprocessor import LayoutPreprocessor
from ..region import LayoutRegion

logger = logging.getLogger(__name__)

_HF_REPO        = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_MODEL_FILENAME = "model.safetensors"          # ultralytics-compatible weights


class PPDocLayoutDetector(BaseLayoutDetector):
    """
    Concrete layout detector backed by PP-DocLayout-V3 running on PyTorch
    through the ultralytics inference API.

    Swap tip: replace this class with any other BaseLayoutDetector subclass
    in pipeline_config.py — the pipeline never references this class directly.
    """

    def __init__(self, config: LayoutConfig) -> None:
        # These are set before super().__init__() calls _load()
        self._model     = None
        self._pre       = LayoutPreprocessor(
            input_size=config.input_size,
            normalize=False,   # ultralytics normalises internally
        )
        self._post      = LayoutPostprocessor(
            score_threshold=config.score_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            containment_overlap=config.containment_overlap,
            keep_discard=config.keep_discard_regions,
        )
        super().__init__(config)

    # ── BaseLayoutDetector contract ────────────────────────────────────────────

    def _load(self) -> None:
        model_path = self._resolve_weights()
        self._model = self._build_model(model_path)

    def _predict(self, image: Image.Image) -> list[LayoutRegion]:
        image = image.convert("RGB")
        raw_regions = self._run_inference(image)
        return self._post.process(raw_regions)

    # ── Model resolution ───────────────────────────────────────────────────────

    def _resolve_weights(self) -> Path:
        if self.config.model_path:
            path = Path(self.config.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Explicit model_path not found: {path}")
            logger.info("[PPDocLayout] Using explicit weights: %s", path)
            return path

        cache = Path(self.config.cache_dir) / _MODEL_FILENAME
        if cache.exists():
            logger.info("[PPDocLayout] Using cached weights: %s", cache)
            return cache

        logger.info("[PPDocLayout] Downloading weights from HuggingFace …")
        downloader = ModelDownloader(cache_root=self.config.cache_dir)
        return Path(downloader.download(
            repo_id=_HF_REPO,
            filename=_MODEL_FILENAME,
        ))

    def _build_model(self, model_path: Path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for PPDocLayoutDetector.\n"
                "Install with: pip install ultralytics"
            ) from exc

        model = YOLO(str(model_path))
        device = self._resolve_device()
        model.to(device)
        logger.info("[PPDocLayout] Model on device: %s", device)
        return model

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_inference(self, image: Image.Image) -> list[LayoutRegion]:
        """Run ultralytics inference and convert raw results to LayoutRegion."""
        results = self._model.predict(
            source=image,
            imgsz=max(self.config.input_size),
            conf=self.config.score_threshold,
            iou=self.config.nms_iou_threshold,
            device=self._resolve_device(),
            verbose=False,
        )

        regions: list[LayoutRegion] = []
        if not results:
            return regions

        result = results[0]
        if result.boxes is None:
            return regions

        boxes  = result.boxes.xyxy.cpu().numpy()   # [N, 4]  x1y1x2y2
        scores = result.boxes.conf.cpu().numpy()   # [N]
        labels = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        for idx, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append(LayoutRegion(
                index=idx,
                label=label_from_index(label_idx),
                score=float(score),
                bbox=[x1, y1, x2, y2],
            ))

        return regions

    # ── Device helper ──────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.config.device