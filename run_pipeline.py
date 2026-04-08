#!/usr/bin/env python3
"""
Minimal entry point to run the OCR pipeline on a file or image.

Usage:
    python run_pipeline.py path/to/document.pdf
    python run_pipeline.py path/to/page.png
    python run_pipeline.py doc.pdf --pages 1
    python run_pipeline.py doc.pdf --pages 2-5
    python run_pipeline.py doc.pdf --pages 1,3,5-8
    python run_pipeline.py doc.pdf --dummy
    python run_pipeline.py doc.pdf --layout-dir debug/
    python run_pipeline.py page.png --layout-out debug/layout.png
    python run_pipeline.py doc.pdf --pages 1-3 --output result.md
    python run_pipeline.py doc.pdf -o report.md --embed-images
    python run_pipeline.py doc.pdf -o report.md   # .md + markdown format → auto figure dir
    python run_pipeline.py doc.pdf -o notes.md --no-embed-images
    python run_pipeline.py doc.pdf --save-images-dir output/figures
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from core import OCRPipeline, PipelineConfig, LayoutConfig, OCRConfig
from core.loader.page_loader import PageLoader, PagesSpec


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_image_save_paths(
    output: Path | None,
    save_images_dir: Path | None,
    embed_images: bool,
) -> tuple[str | None, str | None]:
    """
    Returns (save_images_dir, markdown_image_prefix) for PipelineConfig.

    Explicit ``--save-images-dir`` writes there and uses the same path in links.
    ``--embed-images`` without that flag creates ``<output_stem>_images/`` next
    to ``-o`` and uses that folder name in Markdown so previews resolve.
    """
    if save_images_dir is not None:
        root = save_images_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return str(root), None

    if not embed_images:
        return None, None

    if output is not None:
        out = output.expanduser().resolve()
        folder = out.parent / f"{out.stem}_images"
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder.resolve()), f"{out.stem}_images"

    fallback = (Path.cwd() / "embedded_images").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback), "embedded_images"


def _parse_pages(value: str) -> PagesSpec:
    """
    Convert the --pages CLI string to a PagesSpec.
    Accepts:
        "1"       → single page
        "2-5"     → range
        "1,3,5-8" → mixed
    Validation (bounds checking) happens later inside PageLoader.
    """
    # Let PageLoader._parse_page_string do the real work;
    # we just return the raw string — run_file() accepts str directly.
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the OCR pipeline on a PDF or image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Positional ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--path",
        type=Path,
        help="PDF, image file, or directory of images",
    )

    # ── Page selection ────────────────────────────────────────────────────────
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Pages to process (1-based, PDF / directory only).\n"
            "Examples:\n"
            "  --pages 1          single page\n"
            "  --pages 2-5        range (inclusive)\n"
            "  --pages 1,3,5-8    mixed list and ranges\n"
            "  (omit for all pages)"
        ),
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=("auto", "pdf", "image"),
        default="auto",
        help=(
            "auto  – infer from file extension (default)\n"
            "pdf   – force multi-page PDF path\n"
            "image – force single-image path"
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy layout + OCR models (no downloads, for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for both layout and OCR models: auto | cpu | cuda | cuda:0",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype for the OCR model (default: auto)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("markdown", "json", "text"),
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write result to this file (default: print to stdout)",
    )

    # ── Layout visualisation ──────────────────────────────────────────────────
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save layout overlay images here (PDF / directory mode)",
    )
    parser.add_argument(
        "--layout-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save layout overlay to this file (single-image mode)",
    )

    # ── Figure crops (Markdown ![…](…) and JSON image_path) ────────────────────
    parser.add_argument(
        "--save-images-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Save IMAGE/CHART region crops as PNGs here and reference them in output.\n"
            "Best with --format markdown (or json)."
        ),
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help=(
            "Save crops next to -o as <name>_images/ (or ./embedded_images if no -o).\n"
            "Markdown links use that folder name so previews work next to the .md file."
        ),
    )
    parser.add_argument(
        "--no-embed-images",
        action="store_true",
        help="Turn off automatic figure saving when -o ends with .md (see default behaviour above).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF rasterisation DPI (default: 150)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # ── Validate path ─────────────────────────────────────────────────────────
    if not args.path.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 1

    auto_embed = (
        not args.no_embed_images
        and args.output_format == "markdown"
        and args.output is not None
        and args.output.suffix.lower() == ".md"
        and args.save_images_dir is None
    )
    want_figure_crops = args.embed_images or auto_embed

    save_img_dir, md_img_prefix = _resolve_image_save_paths(
        output=args.output,
        save_images_dir=args.save_images_dir,
        embed_images=want_figure_crops,
    )
    if args.save_images_dir is not None and args.embed_images:
        print(
            "warning: --save-images-dir is set; --embed-images is ignored for the output path",
            file=sys.stderr,
        )

    # ── Build pipeline ────────────────────────────────────────────────────────
    if args.dummy:
        config = PipelineConfig(
            layout=LayoutConfig(detector_name="dummy"),
            ocr=OCRConfig(model_name="dummy"),
            output_format=args.output_format,
            save_images_dir=save_img_dir,
            markdown_image_prefix=md_img_prefix,
        )
    else:
        config = PipelineConfig(
            layout=LayoutConfig(device=args.device),
            ocr=OCRConfig(device=args.device, dtype=args.dtype),
            output_format=args.output_format,
            save_images_dir=save_img_dir,
            markdown_image_prefix=md_img_prefix,
        )

    pipeline = OCRPipeline(config)

    # ── Classify source ───────────────────────────────────────────────────────
    p              = args.path.resolve()
    suf            = p.suffix.lower()
    is_image_file  = p.is_file() and suf in PageLoader.SUPPORTED_IMAGE_EXTS
    is_pdf_or_dir  = p.is_dir() or (p.is_file() and suf == ".pdf")

    # Parse pages spec (stays as string — pipeline accepts str directly)
    pages: PagesSpec = args.pages  # None if not supplied

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        text = _run(
            pipeline=pipeline,
            path=p,
            mode=args.mode,
            is_image_file=is_image_file,
            is_pdf_or_dir=is_pdf_or_dir,
            pages=pages,
            layout_dir=args.layout_dir,
            layout_out=args.layout_out,
            dpi=args.dpi,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # ── Write output ──────────────────────────────────────────────────────────
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.output}")
    else:
        print(text)

    return 0


def _run(
    pipeline:      OCRPipeline,
    path:          Path,
    mode:          str,
    is_image_file: bool,
    is_pdf_or_dir: bool,
    pages:         PagesSpec,
    layout_dir:    Path | None,
    layout_out:    Path | None,
    dpi:           int,
) -> str:
    # ── Explicit mode: pdf ────────────────────────────────────────────────────
    if mode == "pdf":
        if not (path.is_file() and path.suffix.lower() == ".pdf"):
            raise ValueError("--mode pdf requires a .pdf file")
        doc = pipeline.run_file(
            path,
            pages=pages,
            save_layout_dir=layout_dir,
        )
        return doc.merged_text

    # ── Explicit mode: image ──────────────────────────────────────────────────
    if mode == "image":
        if not is_image_file:
            raise ValueError(
                "--mode image requires a single image file "
                f"({', '.join(sorted(PageLoader.SUPPORTED_IMAGE_EXTS))})"
            )
        if pages is not None:
            print(
                "warning: --pages is ignored for single image files",
                file=sys.stderr,
            )
        img    = Image.open(path).convert("RGB")
        result = pipeline.run_image(img, save_layout_path=layout_out)
        return result.formatted

    # ── Auto mode ─────────────────────────────────────────────────────────────
    if is_image_file:
        if pages is not None:
            print(
                "warning: --pages is ignored for single image files",
                file=sys.stderr,
            )
        img    = Image.open(path).convert("RGB")
        result = pipeline.run_image(img, save_layout_path=layout_out)
        return result.formatted

    # PDF or directory
    doc = pipeline.run_file(
        path,
        pages=pages,
        save_layout_dir=layout_dir,
    )
    return doc.merged_text


if __name__ == "__main__":
    raise SystemExit(main())