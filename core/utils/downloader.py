"""
ModelDownloader — thin wrapper around huggingface_hub for weight caching.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Downloads model files from HuggingFace Hub into a local cache directory.

    Usage:
        downloader = ModelDownloader(cache_root="~/.cache/ocr-core")
        path = downloader.download(
            repo_id="PaddlePaddle/PP-DocLayoutV3_safetensors",
            filename="PP-DocLayoutV3.pt",
        )
    """

    def __init__(self, cache_root: str = "~/.cache/ocr-core") -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def download(self, repo_id: str, filename: str) -> str:
        """
        Download a single file from a HuggingFace repo and return its local path.

        The file is stored inside cache_root using huggingface_hub's default
        snapshot directory structure, so repeated calls are instant (cache hit).
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for automatic model downloading.\n"
                "Install with:  pip install huggingface-hub"
            ) from exc

        logger.info("[ModelDownloader] Fetching '%s' from '%s' …", filename, repo_id)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(self.cache_root),
        )
        logger.info("[ModelDownloader] Cached at: %s", local_path)
        return local_path
