"""
GLMOCRModel — runs GLM-OCR locally via HuggingFace transformers.

No HTTP server, no vLLM, no external API.
A threading.Lock ensures the single model instance is safe for concurrent
calls from WorkerPool threads.

torch is imported lazily (inside _load / _recognize) so the module can be
imported on machines without PyTorch installed; the ImportError is only raised
when you actually try to use GLMOCRModel.
"""

from __future__ import annotations

import logging
import threading

from PIL import Image

from ...config.ocr_config import OCRConfig
from ...layout.labels     import DocLabel
from ..base               import BaseOCRModel
from ..prompt_builder     import PromptBuilder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "zai-org/GLM-OCR"


class GLMOCRModel(BaseOCRModel):
    """
    Concrete OCR model backed by GLM-OCR (0.9B multimodal VLM).

    Swap tip: replace this with any other BaseOCRModel in ocr/models/__init__.py.
    The pipeline references only BaseOCRModel.
    """

    def __init__(self, config: OCRConfig) -> None:
        self._model      = None
        self._processor  = None
        self._lock       = threading.Lock()
        self._prompt_bld = PromptBuilder()
        super().__init__(config)

    # ── BaseOCRModel contract ──────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            import torch  # noqa: F401 — validates torch is available
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for GLMOCRModel.\n"
                "Install with:  pip install torch torchvision"
            ) from exc

        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id   = self.config.model_id or _DEFAULT_MODEL_ID
        cache_dir  = str(self.config.cache_dir)
        dtype      = self._resolve_dtype()
        device_map = self._resolve_device_map()

        logger.info("[GLMOCRModel] Loading processor: %s", model_id)
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=self.config.trust_remote_code,
        )

        logger.info(
            "[GLMOCRModel] Loading model (dtype=%s, device_map=%s)",
            dtype, device_map,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model.eval()

    def _recognize(self, crop: Image.Image, label: DocLabel) -> str:
        """
        Thread-safe inference — acquires _lock before entering the model.
        Returns raw decoded text from the model.
        """
        import torch

        messages = self._prompt_bld.build_messages(crop.convert("RGB"), label)

        with self._lock:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            inputs.pop("token_type_ids", None)

            generate_kwargs: dict = dict(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=(self.config.temperature > 0),
            )
            if self.config.temperature > 0:
                generate_kwargs["temperature"] = self.config.temperature

            with torch.inference_mode():
                generated_ids = self._model.generate(**generate_kwargs)

            new_ids  = generated_ids[0][inputs["input_ids"].shape[1]:]
            raw_text = self._processor.decode(new_ids, skip_special_tokens=True)

        return raw_text.strip()

    # ── Device / dtype helpers ─────────────────────────────────────────────────

    def _resolve_device_map(self):
        dev = self.config.device
        if dev == "auto":
            return "auto"
        return {"": dev}

    def _resolve_dtype(self):
        import torch
        if self.config.dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        dtype_map = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype '{self.config.dtype}'. "
                f"Choose from: {list(dtype_map)}"
            )
        return dtype_map[self.config.dtype]
