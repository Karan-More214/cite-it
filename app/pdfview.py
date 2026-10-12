"""Render a PDF page to PNG with the question's key words highlighted."""
from __future__ import annotations

import io

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def render_page(pdf_bytes: bytes, page_no: int, words: list[str], scale: float = 1.7) -> bytes:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        if not 1 <= page_no <= len(pdf):
            raise IndexError("page out of range")
        page = pdf[page_no - 1]
        img: Image.Image = page.render(scale=scale).to_pil().convert("RGBA")
        H = page.get_height()
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        tp = page.get_textpage()
        for w in dict.fromkeys(w for w in words if len(w) >= 3):
            srch = tp.search(w, match_case=False, match_whole_word=False)
            while True:
                hit = srch.get_next()
                if not hit:
                    break
                idx, cnt = hit
                l = b = r = t = None
                for c in range(idx, idx + cnt):
                    x0, y0, x1, y1 = tp.get_charbox(c)
                    l = x0 if l is None else min(l, x0)
                    b = y0 if b is None else min(b, y0)
                    r = x1 if r is None else max(r, x1)
                    t = y1 if t is None else max(t, y1)
                if l is not None:
                    d.rectangle([l * scale, (H - t) * scale, r * scale, (H - b) * scale], fill=(255, 213, 26, 120))
        out = Image.alpha_composite(img, layer).convert("RGB")
        buf = io.BytesIO()
        out.save(buf, "PNG", optimize=True)
        return buf.getvalue()
    finally:
        pdf.close()
