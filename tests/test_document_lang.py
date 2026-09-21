"""
test_document_lang.py — language plumbing + Devanagari fixture/engine gates.

Measured on the fixture (12 pages/script @300dpi, out/doc_p4_devanagari.json):
  clean CER: ne 0.030, hi 0.028 (Latin engine on the same pages: 0.83-0.86)
  degraded:  ne mean 0.094 (pass), hi mean 0.170 (heavy level fails)
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_metrics  # noqa: E402
from document_ocr import Token, available_backends, normalize_lang  # noqa: E402

RAPID_OK = "rapidocr" in available_backends()
try:
    import PySide6  # noqa: F401
    QT_OK = True
except Exception:  # noqa: BLE001
    QT_OK = False


def test_normalize_lang_aliases():
    assert normalize_lang(None) == "default"
    assert normalize_lang("") == "default"
    assert normalize_lang("EN") == "default"
    for code in ("ne", "nep", "nepali", "hi", "hin", "hindi", "Devanagari"):
        assert normalize_lang(code) == "devanagari", code
    assert normalize_lang("klingon") == "default"


def test_devanagari_digits_count_as_digits():
    assert Token("१२३.४५", 90, (0, 0, 10, 10)).has_digits
    assert Token("रकम: १२३", 90, (0, 0, 10, 10)).has_digits
    assert not Token("धन्यवाद", 90, (0, 0, 10, 10)).has_digits


@pytest.mark.skipif(not QT_OK, reason="PySide6 unavailable")
def test_devanagari_fixture_renders_and_gt_matches():
    import doc_data

    img, gt = doc_data.render_devanagari_invoice(seed=1, dpi=300, script="ne")
    assert img.shape[0] > 2000 and img.shape[1] > 1500
    ink = int((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).sum())
    assert ink > 5000, "page must contain rendered glyphs (not tofu/blank)"
    lines = [l for l in gt.splitlines() if l.strip()]
    assert len(lines) >= 20
    assert any("\u0900" <= ch <= "\u097f" for ch in gt), "GT must be Devanagari"
    assert "नेपाल" in gt

    img2, gt2 = doc_data.render_devanagari_invoice(seed=1, dpi=300, script="hi")
    assert "भारत" in gt2 and gt2 != gt


@pytest.mark.skipif(not RAPID_OK or not QT_OK, reason="rapidocr/PySide6 unavailable")
@pytest.mark.parametrize("script", ["ne", "hi"])
def test_devanagari_engine_reads_fixture(script):
    import doc_data
    from document_ocr import ocr_page

    img, gt = doc_data.render_devanagari_invoice(seed=5, dpi=300, script=script)
    try:
        res = ocr_page(img, backend="rapidocr", lang=script)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"devanagari model unavailable: {e}")
    cer = doc_metrics.cer(gt, res.text)
    assert cer < 0.15, f"{script}: clean fixture CER {cer:.3f} (gate < 0.15)"
    latin = ocr_page(img, backend="rapidocr", lang=None)
    assert doc_metrics.cer(gt, latin.text) > 0.5, \
        "the Latin engine must measurably fail on Devanagari (sanity control)"
