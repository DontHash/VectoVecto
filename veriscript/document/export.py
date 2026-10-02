"""
document_export.py — document outputs: searchable PDF, overlay, transcript, JSON.

Permissive stack only:
  * reportlab (BSD-3) writes the searchable PDF: page image + invisible text
    (render mode 3) positioned at OCR bounding boxes. No PyMuPDF (AGPL).
  * the invisible layer and the overlay annotations use the bundled
    Mukta (OFL-1.1, `fonts/`) so Devanagari stays searchable; Helvetica is
    only a logged fallback when no font is found.
  * pypdfium2 is used by tests/callers to read PDFs back; not needed here.

API:
    export_document_outputs(out_dir, stem, image_bgr, ocr_result, dpi=None,
                            make_pdf=True, make_overlay=True, make_txt=True,
                            make_json=True, make_md=True) -> dict of written paths

    write_searchable_pdf(path, image_bgr, tokens, dpi=None)
    write_overlay_png(path, image_bgr, tokens)
    write_transcript(path, result)
    write_transcript_md(path, result, title="Transcript")
    write_combined_transcript_md(path, results, title=stem)
    write_ocr_json(path, result)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from veriscript.document.ocr import OCRResult, Token

FLAG_COLORS = {  # BGR
    "digit_conflict": (0, 0, 255),      # red
    "low_conf": (0, 191, 255),          # amber
    "digit_uncertain": (0, 191, 255),   # amber
    "unknown_word": (0, 191, 255),      # amber: out-of-lexicon Devanagari word
}
OK_COLOR = (0, 180, 0)                  # green

from veriscript.paths import ROOT

_BASE_DIR = ROOT
_UNICODE_FONT_NAME = "VectoDevaUnicode"
_FONT_STATE: Dict[str, Optional[object]] = {"path": None, "tried": False}
_LOG = logging.getLogger("vectovecto.document_export")

_FONT_CANDIDATES = (
    os.path.join(_BASE_DIR, "fonts", "Mukta-Regular.ttf"),
    os.path.join(sys.prefix, "fonts", "Mukta-Regular.ttf"),
    r"C:\Windows\Fonts\mangal.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    # Noto Sans Devanagari last: reportlab's subsetter drops ASCII letters
    # from it (measured), so it is only a last resort.
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
)


def _reset_font_cache() -> None:
    """Test hook: forget the resolved font so candidates can be re-probed."""
    _FONT_STATE.update({"path": None, "tried": False})


def unicode_font_path() -> Optional[str]:
    """Path of a Devanagari-capable TTF, or None (logged once).

    Search order: the bundled OFL font (`fonts/`), the wheel data-files
    location (`sys.prefix/fonts`), then OS fallbacks (Windows Mangal, Linux
    Noto/Lohit — the same families `doc_data` renders fixtures with).
    """
    if not _FONT_STATE["tried"]:
        _FONT_STATE["tried"] = True
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                _FONT_STATE["path"] = path
                break
        if not _FONT_STATE["path"]:
            _LOG.warning(
                "no Devanagari font found; the PDF text layer falls back to "
                "Helvetica and Devanagari will not be searchable")
    return _FONT_STATE["path"]  # type: ignore[return-value]


def unicode_pdf_font() -> Optional[str]:
    """Registered reportlab font name for the text layer, or None."""
    path = unicode_font_path()
    if not path:
        return None
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    try:
        if _UNICODE_FONT_NAME not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(_UNICODE_FONT_NAME, path))
        return _UNICODE_FONT_NAME
    except Exception as e:  # noqa: BLE001 - a font problem must not break export
        _LOG.warning("could not register %s: %s", path, e)
        return None


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
    font = unicode_pdf_font() or "Helvetica"
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
            t.setFont(font, size)
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


def _draw_annotations(canvas: np.ndarray,
                      anns: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]]
                      ) -> np.ndarray:
    """Draw `[(text, (x, y), color_bgr)]` above the boxes.

    PIL + the Devanagari font renders `alt_text` glyphs (cv2's Hershey fonts
    cannot); cv2 remains the fallback when no font is available.
    """
    path = unicode_font_path()
    if path:
        try:
            from PIL import ImageDraw, ImageFont
            font = ImageFont.truetype(path, 14)
            img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            for text, (x, y), color in anns:
                draw.text((x, y), text, font=font,
                          fill=(int(color[2]), int(color[1]), int(color[0])))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:  # noqa: BLE001 - annotations are best-effort
            _LOG.warning("PIL annotation failed (%s); falling back to cv2", e)
    out = canvas.copy()
    for text, (x, y), color in anns:
        cv2.putText(out, text, (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    return out


def write_overlay_png(path: str, image_bgr: np.ndarray, tokens: List[Token]) -> str:
    """Boxes: green ok, amber uncertain, red conflict (with the alt reading)."""
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        color = _tok_color(tok)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
    anns: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        color = _tok_color(tok)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        if "digit_conflict" in tok.flags and tok.alt_text:
            anns.append((f"! {tok.alt_text}", (x0, max(0, y0 - 18)), color))
    if anns:
        canvas = _draw_annotations(canvas, anns)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


def write_transcript(path: str, result: OCRResult) -> str:
    lines = [t.text for t in result.tokens if t.text]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def transcript_md_text(result: OCRResult) -> str:
    """Markdown body for one page: provenance line, transcript, review table.

    Deterministic (no timestamps): the same OCR result always produces the
    same bytes. The review table is omitted when nothing was flagged.
    """
    from veriscript.document.ocr import review_queue

    review = review_queue(result.tokens)
    low = sum(1 for t in result.tokens if "low_conf" in t.flags)
    conflicts = sum(1 for t in result.tokens if "digit_conflict" in t.flags)

    body = [
        f"*{result.backend} · {len(result.tokens)} tokens · {low} "
        f"low-confidence · {conflicts} digit conflicts · "
        f"{len(review)} to review*",
        "",
    ]
    body.extend(t.text for t in result.tokens if t.text)
    if review:
        body += [
            "",
            "## Review queue",
            "",
            "| token | flags | alternative | conf |",
            "|---|---|---|---|",
        ]
        for tok in review:
            text = (tok.text or "").replace("|", "\\|").replace("\n", " ")
            alt = (tok.alt_text or "").replace("|", "\\|").replace("\n", " ")
            alt_cell = f"`{alt}`" if alt else "—"
            body.append(f"| `{text}` | {', '.join(tok.flags)} | {alt_cell} | "
                        f"{tok.conf:.0f} |")
    return "\n".join(body) + "\n"


def write_transcript_md(path: str, result: OCRResult,
                        title: Optional[str] = "Transcript") -> str:
    """Markdown transcript for one page (transcript + review-queue table)."""
    text = transcript_md_text(result)
    if title:
        text = f"# {title}\n\n{text}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_combined_transcript_md(path: str, results, title: str) -> str:
    """Multi-page Markdown: one `## Page N` section per OCRResult."""
    parts = [f"# {title}", ""]
    for idx, result in enumerate(results):
        parts.append(f"## Page {idx + 1}")
        parts.append("")
        parts.append(transcript_md_text(result))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def write_ocr_json(path: str, result: OCRResult) -> str:
    from veriscript.document.ocr import review_queue, token_risk

    review = review_queue(result.tokens)
    index_of = {id(t): i for i, t in enumerate(result.tokens)}
    payload = {
        "backend": result.backend,
        "meta": {k: v for k, v in result.meta.items()},
        "review": [
            {
                "index": index_of[id(t)],
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
                "original_text": t.orig_text,
                "text_source": t.text_source,
                "corrected_by": t.corrected_by,
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
                            make_md: bool = True,
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
    if make_md:
        written["md"] = write_transcript_md(os.path.join(out_dir, f"{stem}.md"), result)
    if make_json:
        written["json"] = write_ocr_json(os.path.join(out_dir, f"{stem}.json"), result)
    return written


if __name__ == "__main__":
    import sys
    from veriscript.core import data as doc_data
    from veriscript.core.degradation import degrade_page
    from veriscript.document.ocr import ocr_page

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
