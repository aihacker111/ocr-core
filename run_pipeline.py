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
from core import OCRPipeline, PipelineConfig, LayoutConfig, OCRConfig


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
    pipeline = OCRPipeline()

    text = pipeline.run_file_to_string(args.path)

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print("Wrote:", args.output)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
