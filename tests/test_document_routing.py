"""
test_document_routing.py — page classifier + SmartUpscaler document mode.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from smart_upscaler import SmartUpscaler, is_document, page_likeness  # noqa: E402


def test_page_likeness_separates_document_from_photo_like():
    page, _gt = doc_data.render_synthetic_invoice(seed=81, dpi=150)
    assert page_likeness(page) >= 0.45, "invoice page must classify as document"

    rng = np.random.default_rng(0)
    noise = (rng.random((400, 400, 3)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), 3.0)
    assert page_likeness(noise) < 0.45
    assert page_likeness(blurred) < 0.45


def test_auto_routes_document_without_loading_photo_engine(monkeypatch):
    import smart_upscaler

    page, _gt = doc_data.render_synthetic_invoice(seed=82, dpi=150)
    monkeypatch.setattr(smart_upscaler, "is_document", lambda img, threshold=0.45: True)
    upscaler = SmartUpscaler()
    out = upscaler.upscale(page, scale=1, mode="auto")
    assert upscaler.engine is None, "photo engine must not be loaded for a document"
    assert out.shape == page.shape
    assert out.dtype == np.uint8


def test_document_mode_returns_cleaned_display():
    page, _gt = doc_data.render_synthetic_invoice(seed=83, dpi=150)
    # add a lighting gradient so the display visibly changes
    h, w = page.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shaded = np.clip(page.astype(np.float32) * (1.0 - 0.35 * xx / w)[:, :, None], 0, 255)
    shaded = shaded.astype(np.uint8)

    upscaler = SmartUpscaler()
    out = upscaler.upscale(shaded, scale=1, mode="document")
    assert out.shape == shaded.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, shaded), "restore must change the page"

    out2x = upscaler.upscale(shaded, scale=2, mode="document")
    assert out2x.shape[0] == shaded.shape[0] * 2
    assert out2x.shape[1] == shaded.shape[1] * 2


def test_is_document_threshold_default():
    page, _gt = doc_data.render_synthetic_invoice(seed=84, dpi=150)
    assert is_document(page) is True


def test_pipeline_primary_stream_follows_recommended_table(monkeypatch):
    import document_ocr
    import document_pipeline as dp
    from document_ocr import OCRResult

    if "tesseract" not in document_ocr.available_backends():
        import pytest

        pytest.skip("tesseract binary not installed (external dependency)")

    calls = []

    def fake_ocr_page(img, backend="rapidocr", **kw):
        calls.append((backend, id(img)))
        return OCRResult(text="", tokens=[], backend=backend, meta={})

    monkeypatch.setattr(dp, "ocr_page", fake_ocr_page)
    monkeypatch.setattr(dp, "pick_backend", lambda b=None: b or "rapidocr")
    monkeypatch.setattr(dp, "compare_digit_streams", lambda a, b: 0)

    page = np.full((80, 200, 3), 255, dtype=np.uint8)

    res = dp.run_document_pipeline(page, backend="tesseract")
    assert res.meta["primary_stream"] == "restored"
    assert calls[0][1] != id(page), "tesseract must OCR the restored display first"

    calls.clear()
    res = dp.run_document_pipeline(page, backend="rapidocr")
    assert res.meta["primary_stream"] == "raw"
    assert calls[0][1] == id(page), "rapidocr must OCR the raw pixels first"
