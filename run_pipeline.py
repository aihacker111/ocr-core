#!/usr/bin/env python3
"""
Minimal entry point to run the OCR pipeline on a file or image.

Usage:
    python run_pipeline.py path/to/document.pdf
    python run_pipeline.py path/to/page.png
    python run_pipeline.py path/to/doc.pdf --dummy   # no model weights

From another directory, use the repo root as cwd or PYTHONPATH.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root must be on path (Colab: cd /content/ocr-core before running).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from core import OCRPipeline, PipelineConfig, LayoutConfig, OCRConfig
from core.loader.page_loader import PageLoader


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR pipeline on a PDF or image.")
    parser.add_argument(
        "path",
        type=Path,
        help="PDF, image file, or directory of images",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy layout + OCR (no downloads, for smoke tests)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write merged text to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print("error: path does not exist:", args.path, file=sys.stderr)
        return 1

    if args.dummy:
        pipeline = OCRPipeline(
            PipelineConfig(
                layout=LayoutConfig(detector_name="dummy"),
                ocr=OCRConfig(model_name="dummy"),
            )
        )
    else:
        pipeline = OCRPipeline()

    # Single image file → public run_image API (same OCR path as run_file for one page).
    p = args.path.resolve()
    if p.is_file() and p.suffix.lower() in PageLoader.SUPPORTED_IMAGE_EXTS:
        img = Image.open(p).convert("RGB")
        text = pipeline.run_image(img).formatted
    else:
        text = pipeline.run_file_to_string(p)

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print("Wrote:", args.output)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
