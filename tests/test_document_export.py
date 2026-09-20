"""
test_document_export.py — outputs: searchable PDF, overlay, transcript, JSON.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from degradation_document import degrade_page  # noqa: E402
from document_export import export_document_outputs, write_overlay_png, write_searchable_pdf  # noqa: E402
from document_ocr import Token, available_backends, ocr_page  # noqa: E402


def test_searchable_pdf_pagesize_matches_dpi(tmp_path):
    img = np.full((300, 200, 3), 255, dtype=np.uint8)
    toks = [Token(text="HELLO", conf=99, bbox=(10, 10, 120, 40), granularity="word")]
    path = write_searchable_pdf(str(tmp_path / "t.pdf"), img, toks, dpi=150)
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    w, h = doc[0].get_size()
    assert abs(w - 200 * 72 / 150) < 1
    assert abs(h - 300 * 72 / 150) < 1
    text = doc[0].get_textpage().get_text_range()
    assert "HELLO" in text


def test_overlay_marks_conflicts_red(tmp_path):
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    toks = [
        Token(text="1200", conf=90, bbox=(20, 20, 200, 60), granularity="word",
              flags=["digit_conflict"], alt_text="1260"),
        Token(text="Vendor", conf=99, bbox=(20, 65, 200, 95), granularity="word"),
    ]
    path = write_overlay_png(str(tmp_path / "o.png"), img, toks)
    overlay = cv2.imread(path)
    assert overlay.shape == img.shape
    conflict_region = overlay[18:62, 18:202]
    red = ((conflict_region[:, :, 2] > 200) & (conflict_region[:, :, 0] < 80)).sum()
    assert red > 20, "conflict box must be red (BGR)"
    ok_region = overlay[63:97, 18:202]
    green = ((ok_region[:, :, 1] > 120) & (ok_region[:, :, 2] < 80)).sum()
    assert green > 20, "agreeing token must be green (BGR)"


def test_ocr_json_review_queue_is_ranked(tmp_path):
    from document_export import write_ocr_json
    from document_ocr import OCRResult, review_queue, token_risk

    toks = [
        Token(text="note", conf=50, bbox=(0, 100, 40, 120), granularity="word",
              flags=["low_conf"]),
        Token(text="TOTAL 1200.00", conf=90, bbox=(0, 0, 200, 30), granularity="line",
              flags=["digit_conflict"], alt_text="TOTAL 1260.00"),
        Token(text="INVOICE", conf=99, bbox=(0, 200, 100, 220), granularity="word"),
    ]
    result = OCRResult(text="x", tokens=toks, backend="rapidocr")
    path = write_ocr_json(str(tmp_path / "r.json"), result)
    payload = json.load(open(path, encoding="utf-8"))
    assert len(payload["review"]) == 2, "only flagged tokens are in the queue"
    assert payload["review"][0]["text"] == "TOTAL 1200.00"
    assert payload["review"][0]["risk"] > payload["review"][1]["risk"]
    assert review_queue(toks, top_k=1)[0].text == "TOTAL 1200.00"
    assert token_risk(toks[2]) == 0.0


@pytest.mark.skipif("rapidocr" not in available_backends(), reason="rapidocr unavailable")
def test_export_document_outputs_end_to_end(tmp_path):
    page, _gt = doc_data.render_synthetic_invoice(seed=31, dpi=200)
    degraded = degrade_page(page, seed=3101, level="mild")
    result = ocr_page(degraded, backend="rapidocr")
    out = export_document_outputs(str(tmp_path), "inv", degraded, result)
    assert set(out.keys()) == {"pdf", "overlay", "txt", "json"}
    for path in out.values():
        assert os.path.getsize(path) > 0

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(out["pdf"])
    text = doc[0].get_textpage().get_text_range().upper()
    assert "INVOICE" in text
    assert any(k in text for k in ("TOTAL", "INVOICE NO", "DATE"))

    overlay = cv2.imread(out["overlay"])
    assert overlay.shape == degraded.shape

    lines = open(out["txt"], encoding="utf-8").read().splitlines()
    assert len(lines) >= 5

    payload = json.load(open(out["json"], encoding="utf-8"))
    assert payload["backend"] == "rapidocr"
    assert "review" in payload, "JSON must carry the human review queue"
    assert len(payload["tokens"]) == len(result.tokens)
    first = payload["tokens"][0]
    assert {"text", "conf", "bbox", "flags"} <= set(first.keys())
