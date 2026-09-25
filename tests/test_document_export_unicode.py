"""
test_document_export_unicode.py — Devanagari text layer + overlay annotations (W1).

The PDF text layer must extract as searchable Devanagari (the flagship claim);
the bundled Mukta (OFL-1.1, `fonts/`) is registered for both the invisible
layer and the review-overlay annotations, with Helvetica only as a logged
fallback. Measured before choosing it: Noto Sans Devanagari loses ASCII
letters in reportlab's subsetter, Mukta keeps Latin + Devanagari + conjuncts.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import document_export as de  # noqa: E402
from document_export import (unicode_font_path, unicode_pdf_font,  # noqa: E402
                             write_overlay_png, write_searchable_pdf,
                             write_searchable_pdf_pages)
from document_ocr import Token  # noqa: E402


def _extract(path: str) -> str:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    try:
        return "\n".join(doc[i].get_textpage().get_text_range()
                         for i in range(len(doc)))
    finally:
        doc.close()


def test_devanagari_pdf_text_layer_round_trip(tmp_path):
    img = np.full((200, 700, 3), 255, np.uint8)
    toks = [Token(text="नेपाल सरकार १२३", conf=90, bbox=(20, 50, 420, 90)),
            Token(text="रु. १,२३४.५०", conf=90, bbox=(20, 110, 300, 150))]
    path = write_searchable_pdf(str(tmp_path / "ne.pdf"), img, toks, dpi=150)
    text = _extract(path)
    assert "नेपाल सरकार १२३" in text
    assert "रु. १,२३४.५०" in text


def test_multi_page_devanagari_round_trip(tmp_path):
    pages = []
    for text in ("क्ष त्र ज्ञ", "हाम्रो नेपाल"):
        img = np.full((300, 500, 3), 255, np.uint8)
        pages.append((img, [Token(text=text, conf=90, bbox=(10, 10, 400, 50))],
                      150))
    path = write_searchable_pdf_pages(str(tmp_path / "multi.pdf"), pages)
    text = _extract(path)
    assert "क्ष त्र ज्ञ" in text
    assert "हाम्रो नेपाल" in text


def test_latin_text_still_extracts_with_unicode_font(tmp_path):
    img = np.full((150, 500, 3), 255, np.uint8)
    toks = [Token(text="TOTAL 1200.00", conf=90, bbox=(10, 10, 300, 40))]
    path = write_searchable_pdf(str(tmp_path / "en.pdf"), img, toks, dpi=150)
    assert "TOTAL 1200.00" in _extract(path)


def test_overlay_renders_devanagari_alt_text(tmp_path):
    img = np.full((120, 400, 3), 255, np.uint8)
    toks = [Token(text="कुल जम्मा १२३", conf=90, bbox=(20, 50, 350, 90),
                  flags=["digit_conflict"], alt_text="कुल जम्मा १२८")]
    path = write_overlay_png(str(tmp_path / "ov.png"), img, toks)
    out = cv2.imread(path)
    assert out.shape == img.shape
    annotation = out[8:46, 20:250]
    assert (annotation < 200).any(), \
        "the Devanagari alt reading must draw above the box"


def test_font_resolution_and_helvetica_fallback(tmp_path, monkeypatch):
    de._reset_font_cache()
    assert unicode_font_path() is not None
    assert unicode_pdf_font() == "VectoDevaUnicode"

    monkeypatch.setattr(de, "_FONT_CANDIDATES",
                        (str(tmp_path / "missing.ttf"),))
    de._reset_font_cache()
    assert de.unicode_font_path() is None
    assert de.unicode_pdf_font() is None

    img = np.full((100, 300, 3), 255, np.uint8)
    toks = [Token(text="HELLO", conf=90, bbox=(10, 10, 100, 40))]
    path = write_searchable_pdf(str(tmp_path / "fallback.pdf"), img, toks,
                                dpi=150)
    assert "HELLO" in _extract(path)
    de._reset_font_cache()


def test_draw_annotations_cv2_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(de, "_FONT_CANDIDATES",
                        (str(tmp_path / "missing.ttf"),))
    de._reset_font_cache()
    img = np.full((80, 300, 3), 255, np.uint8)
    out = de._draw_annotations(img, [("! 123", (10, 10), (0, 0, 255))])
    assert out.shape == img.shape
    assert (out != img).any()
    de._reset_font_cache()
