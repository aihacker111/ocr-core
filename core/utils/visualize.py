"""
LayoutVisualizer — draws bounding boxes and labels onto a PIL image
after layout detection.

Each DocLabel group gets a distinct colour so it's easy to distinguish
text blocks, tables, figures, formulas, and metadata at a glance.

Usage (standalone):
    from ocr_core.utils.layout_visualizer import LayoutVisualizer
    vis = LayoutVisualizer()
    annotated = vis.draw(image, regions)
    annotated.save("layout_debug.png")

Usage (via pipeline):
    pipeline = OCRPipeline(config)
    doc = pipeline.run_file("document.pdf", save_layout_dir="debug/")
    # → saves  debug/page_0000_layout.png, debug/page_0001_layout.png, …
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from ..layout.labels import DocLabel
from ..layout.region import LayoutRegion

_LABEL_COLOURS: dict[DocLabel, tuple[int, int, int]] = {
    # Blues — textual
    DocLabel.DOC_TITLE:          (30,  100, 220),
    DocLabel.PARAGRAPH_TITLE:    (60,  140, 240),
    DocLabel.TEXT:               (90,  160, 255),
    DocLabel.ABSTRACT:           (50,  120, 200),
    DocLabel.CONTENT:            (80,  150, 235),
    DocLabel.ASIDE_TEXT:         (110, 170, 255),
    DocLabel.REFERENCE:          (100, 170, 255),
    DocLabel.REFERENCE_CONTENT:  (115, 175, 255),
    DocLabel.FOOTNOTE:           (70,  145, 225),
    DocLabel.ALGORITHM:          (20,  80,  180),
    # Greens — graphical
    DocLabel.IMAGE:              (40,  180,  80),
    DocLabel.FIGURE_TITLE:       (80,  200, 120),
    DocLabel.CHART:              (30,  160,  60),
    DocLabel.VISION_FOOTNOTE:    (60,  190, 100),
    # Orange — table
    DocLabel.TABLE:              (230, 130,  20),
    # Purples — formula
    DocLabel.FORMULA:            (170,  60, 220),
    DocLabel.FORMULA_NUMBER:     (190,  90, 240),
    # Greys — metadata
    DocLabel.HEADER:             (130, 130, 130),
    DocLabel.FOOTER:             (130, 130, 130),
    DocLabel.NUMBER:             (150, 150, 150),
    DocLabel.SEAL:               (110, 110, 110),
    # Red — noise
    DocLabel.ABANDON:            (220,  50,  50),
}

class LayoutVisualizer:
    def __init__(self, show_score=True, show_index=True, font_path=None):
        self.show_score = show_score
        self.show_index = show_index
        self._font_path = font_path

    def draw(self, image: Image.Image, regions: list[LayoutRegion]) -> Image.Image:
        canvas  = image.convert("RGBA").copy()
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        font_size = max(10, min(28, int(image.size[1] * 0.018)))
        font      = self._load_font(font_size)
        stroke_w  = max(2, min(6, min(image.size) // 400))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_canvas  = ImageDraw.Draw(canvas)
        for region in regions:
            r, g, b = _LABEL_COLOURS.get(region.label, (180, 180, 180))
            x1, y1, x2, y2 = region.bbox
            draw_overlay.rectangle([x1,y1,x2,y2], fill=(r,g,b,35))
            draw_canvas.rectangle([x1,y1,x2,y2], outline=(r,g,b,255), width=stroke_w)
            tag = " ".join(filter(None, [
                f"#{region.index}" if self.show_index else "",
                region.label.value,
                f"{region.score:.2f}" if self.show_score else "",
            ]))
            tx, ty = x1 + stroke_w, max(0, y1 - font_size - stroke_w - 2)
            try:
                tb = draw_canvas.textbbox((tx, ty), tag, font=font)
            except AttributeError:
                tw, th = draw_canvas.textsize(tag, font=font)
                tb = (tx, ty, tx+tw, ty+th)
            draw_canvas.rectangle([tb[0]-3,tb[1]-3,tb[2]+3,tb[3]+3], fill=(r,g,b))
            draw_canvas.text((tx,ty), tag, font=font, fill=(255,255,255),
                             stroke_width=1, stroke_fill=(0,0,0))
        return Image.alpha_composite(canvas, overlay).convert("RGB")

    def save(self, image, regions, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.draw(image, regions).save(str(out))
        return out

    def _load_font(self, size):
        if self._font_path:
            try: return ImageFont.truetype(self._font_path, size)
            except: pass
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                  "/System/Library/Fonts/Helvetica.ttc",
                  "C:/Windows/Fonts/arial.ttf"]:
            try: return ImageFont.truetype(p, size)
            except: continue
        return ImageFont.load_default()