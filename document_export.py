"""
document_export.py — document outputs: searchable PDF, overlay, transcript, JSON.

Permissive stack only:
  * reportlab (BSD-3) writes the searchable PDF: page image + invisible text
    (render mode 3) positioned at OCR bounding boxes. No PyMuPDF (AGPL).
  * pypdfium2 is used by tests/callers to read PDFs back; not needed here.

API:
    export_document_outputs(out_dir, stem, image_bgr, ocr_result, dpi=None,
                            make_pdf=True, make_overlay=True, make_txt=True,
                            make_json=True) -> dict of written paths

    write_searchable_pdf(path, image_bgr, tokens, dpi=None)
    write_overlay_png(path, image_bgr, tokens)
    write_transcript(path, result)
    write_ocr_json(path, result)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from document_ocr import OCRResult, Token

FLAG_COLORS = {  # BGR
    "digit_conflict": (0, 0, 255),      # red
    "low_conf": (0, 191, 255),          # amber
    "digit_uncertain": (0, 191, 255),   # amber
}
OK_COLOR = (0, 180, 0)                  # green


def _estimate_dpi(img: np.ndarray, dpi: Optional[int] = None) -> int:
    """A4-ish page: 300 DPI when the long side is >= 2000 px, else 150."""
    if dpi:
        return dpi
    return 300 if max(img.shape[:2]) >= 2000 else 150


def write_searchable_pdf_pages(path: str, pages, title: Optional[str] = None) -> str:
    """Multi-page searchable PDF: one (image_bgr, tokens, dpi|None) per page.

    Each page keeps its own size (derived from pixels + dpi), so mixed-size
    inputs stay honest. Per-page artifacts are unaffected; this is the
    combined output for multi-page PDF inputs (P6).
    """
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    c = canvas.Canvas(path)
    c.setTitle(title or os.path.splitext(os.path.basename(path))[0])
    for image_bgr, tokens, dpi in pages:
        dpi = _estimate_dpi(image_bgr, dpi)
        h, w = image_bgr.shape[:2]
        pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
        px2pt = 72.0 / dpi
        c.setPageSize((pw, ph))
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        c.drawImage(ImageReader(Image.fromarray(rgb)), 0, 0, width=pw, height=ph)
        for tok in tokens:
            text = (tok.text or "").strip()
            if not text:
                continue
            x0, y0, x1, y1 = tok.bbox
            size = max(4.0, (y1 - y0) * px2pt * 0.85)
            t = c.beginText()
            t.setTextRenderMode(3)  # invisible but selectable/extractable
            t.setFont("Helvetica", size)
            t.setTextOrigin(x0 * px2pt, ph - y1 * px2pt)
            t.textOut(text)
            c.drawText(t)
        c.showPage()
    c.save()
    return path


def write_searchable_pdf(path: str, image_bgr: np.ndarray, tokens: List[Token],
                         dpi: Optional[int] = None) -> str:
    """Image page + invisible, selectable text at token boxes."""
    return write_searchable_pdf_pages(path, [(image_bgr, tokens, dpi)])


def _tok_color(tok: Token):
    for flag, color in FLAG_COLORS.items():
        if flag in tok.flags:
            return color
    return OK_COLOR


def write_overlay_png(path: str, image_bgr: np.ndarray, tokens: List[Token]) -> str:
    """Boxes: green ok, amber uncertain, red conflict (with the alt reading)."""
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        color = _tok_color(tok)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        color = _tok_color(tok)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        if "digit_conflict" in tok.flags and tok.alt_text:
            cv2.putText(canvas, f"! {tok.alt_text}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


def write_transcript(path: str, result: OCRResult) -> str:
    lines = [t.text for t in result.tokens if t.text]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_ocr_json(path: str, result: OCRResult) -> str:
    from document_ocr import review_queue, token_risk

    review = review_queue(result.tokens)
    payload = {
        "backend": result.backend,
        "meta": {k: v for k, v in result.meta.items()},
        "review": [
            {
                "text": t.text,
                "bbox": list(t.bbox),
                "conf": round(t.conf, 2),
                "risk": round(token_risk(t), 2),
                "flags": t.flags,
                "alt_text": t.alt_text,
            }
            for t in review
        ],
        "tokens": [
            {
                "text": t.text,
                "conf": round(t.conf, 2),
                "bbox": list(t.bbox),
                "granularity": t.granularity,
                "flags": t.flags,
                "alt_text": t.alt_text,
                "repass_text": t.repass_text,
                "repass_conf": t.repass_conf,
            }
            for t in result.tokens
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def export_document_outputs(out_dir: str, stem: str, image_bgr: np.ndarray,
                            result: OCRResult, dpi: Optional[int] = None,
                            make_pdf: bool = True, make_overlay: bool = True,
                            make_txt: bool = True, make_json: bool = True,
                            overlay_source: Optional[np.ndarray] = None) -> Dict[str, str]:
    """Write the product outputs. `overlay_source` lets the caller draw boxes on
    the restored/display image instead of the raw input."""
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    if make_pdf:
        written["pdf"] = write_searchable_pdf(os.path.join(out_dir, f"{stem}.pdf"),
                                              image_bgr, result.tokens, dpi=dpi)
    if make_overlay:
        written["overlay"] = write_overlay_png(os.path.join(out_dir, f"{stem}_overlay.png"),
                                               overlay_source if overlay_source is not None else image_bgr,
                                               result.tokens)
    if make_txt:
        written["txt"] = write_transcript(os.path.join(out_dir, f"{stem}.txt"), result)
    if make_json:
        written["json"] = write_ocr_json(os.path.join(out_dir, f"{stem}.json"), result)
    return written


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import doc_data
    from degradation_document import degrade_page
    from document_ocr import ocr_page

    page, gt = doc_data.render_synthetic_invoice(seed=21, dpi=300)
    deg = degrade_page(page, seed=2101, level="medium")
    res = ocr_page(deg, backend="rapidocr")
    out = export_document_outputs("out/api_check/export_demo", "invoice", deg, res)
    for kind, path in out.items():
        print(f"{kind}: {os.path.getsize(path)} bytes -> {path}")
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(out["pdf"])
    text = doc[0].get_textpage().get_text_range()
    print("pdf text sample:", text[:80].replace("\r\n", " | "))
    print("document_export OK")
