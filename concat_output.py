#!/usr/bin/env python3
"""
concat_output.py — standalone script

Renders markdown to a real captured image (via HTML → WeasyPrint → PNG),
then concatenates side-by-side:
    [original image] | [layout detection image] | [rendered markdown image]
→ saves as concat_output.png

**Images inside Markdown:** WeasyPrint must know where relative ``src`` paths
are resolved from. Pass ``base_url`` = directory containing the ``.md`` file
(``MARKDOWN_FILE.parent.as_uri()``). Without this, ``![](folder/img.png)``
renders as broken images in the PDF/PNG.

Dependencies:
    pip install Pillow markdown weasyprint pymupdf
"""

from __future__ import annotations

from pathlib import Path

import markdown
from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these values
# ══════════════════════════════════════════════════════════════════════════════

ORIGINAL_IMAGE = Path("/content/original_images.png")
LAYOUT_IMAGE   = Path("/content/page_0009_layout.png")
MARKDOWN_FILE  = Path("/content/out.md")
OUTPUT_FILE    = Path("concat_output_2.png")

# If your MD uses paths relative to another folder, set explicitly (file URI).
# Example: Path("/content").resolve().as_uri()
MD_BASE_URL: str | None = None  # None → MARKDOWN_FILE.parent

PANEL_HEIGHT   = 1200
MD_PAGE_WIDTH  = 900
MD_SCALE       = 2

# ══════════════════════════════════════════════════════════════════════════════


PANEL_GAP      = 20
HEADER_HEIGHT  = 44
HEADER_BG      = (50, 50, 50)
HEADER_FG      = (255, 255, 255)
HEADER_FONT_SZ = 16

_MD_CSS = """
    @page { size: 900px 1800px; margin: 0; }
    body {
        font-family: Arial, sans-serif;
        font-size: 15px;
        line-height: 1.7;
        color: #1e1e1e;
        background: #ffffff;
        padding: 32px 40px;
        width: 820px;
        box-sizing: border-box;
    }
    h1 { font-size: 24px; color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 6px; margin-top: 16px; }
    h2 { font-size: 19px; color: #283593; border-bottom: 1px solid #c5cae9; padding-bottom: 4px; margin-top: 14px; }
    h3 { font-size: 15px; color: #303f9f; }
    p  { margin: 8px 0; }
    img { max-width: 100%; height: auto; display: block; margin: 12px 0; }
    blockquote {
        border-left: 4px solid #7986cb;
        background: #e8eaf6;
        margin: 12px 0;
        padding: 8px 16px;
        color: #37474f;
        border-radius: 0 4px 4px 0;
    }
    code {
        background: #f3f4f6;
        border: 1px solid #e0e0e0;
        border-radius: 3px;
        padding: 1px 5px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #c62828;
    }
    pre {
        background: #263238;
        color: #cfd8dc;
        border-radius: 6px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.5;
    }
    pre code {
        background: none;
        border: none;
        color: inherit;
        padding: 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 14px;
    }
    th {
        background: #3949ab;
        color: white;
        padding: 8px 12px;
        text-align: left;
    }
    td {
        border: 1px solid #c5cae9;
        padding: 7px 12px;
    }
    tr:nth-child(even) td { background: #e8eaf6; }
    em  { color: #555; }
    hr  { border: none; border-top: 1px solid #e0e0e0; margin: 16px 0; }
"""


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    w, h = img.size
    new_w = int(w * height / h)
    return img.resize((new_w, height), Image.LANCZOS)


def _add_label(img: Image.Image, label: str) -> Image.Image:
    font = _load_font(HEADER_FONT_SZ)
    total = Image.new("RGB", (img.width, img.height + HEADER_HEIGHT), HEADER_BG)
    draw = ImageDraw.Draw(total)
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw, _ = draw.textsize(label, font=font)
    tx = (img.width - tw) // 2
    ty = (HEADER_HEIGHT - HEADER_FONT_SZ) // 2
    draw.text((tx, ty), label, font=font, fill=HEADER_FG)
    total.paste(img, (0, HEADER_HEIGHT))
    return total


def build_image_panel(path: Path, label: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = _resize_to_height(img, PANEL_HEIGHT)
    return _add_label(img, label)


def build_markdown_panel(md_path: Path, label: str) -> Image.Image:
    from weasyprint import HTML as WeasyprintHTML

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF→image conversion.\n"
            "Install with:  pip install pymupdf"
        ) from exc

    md_path = md_path.resolve()
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{_MD_CSS}</style>
</head>
<body>{html_body}</body>
</html>"""

    # Critical: relative ![](out_images/foo.png) resolves against this URL.
    base_url = MD_BASE_URL if MD_BASE_URL is not None else md_path.parent.as_uri()
    pdf_bytes = WeasyprintHTML(string=full_html, base_url=base_url).write_pdf()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(MD_SCALE, MD_SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    img = _resize_to_height(img, PANEL_HEIGHT)
    return _add_label(img, label)


def concat_panels(panels: list[Image.Image]) -> Image.Image:
    total_w = sum(p.width for p in panels) + PANEL_GAP * (len(panels) - 1)
    total_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (total_w, total_h), (220, 220, 220))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + PANEL_GAP
    return canvas


def main() -> None:
    for path, name in [
        (ORIGINAL_IMAGE, "original image"),
        (LAYOUT_IMAGE, "layout image"),
        (MARKDOWN_FILE, "markdown file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    print(f"Building panels (height={PANEL_HEIGHT}px) …")
    print("  [1/3] Original image …")
    p1 = build_image_panel(ORIGINAL_IMAGE, "Original")

    print("  [2/3] Layout detection image …")
    p2 = build_image_panel(LAYOUT_IMAGE, "Layout Detection")

    print("  [3/3] Rendering markdown → HTML → PDF → image …")
    print(f"       WeasyPrint base_url = {MD_BASE_URL or MARKDOWN_FILE.resolve().parent.as_uri()}")
    p3 = build_markdown_panel(MARKDOWN_FILE, "OCR Output (Markdown)")

    print("Concatenating …")
    result = concat_panels([p1, p2, p3])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(OUTPUT_FILE))
    print(f"Saved → {OUTPUT_FILE}  ({result.width}×{result.height} px)")


if __name__ == "__main__":
    main()
