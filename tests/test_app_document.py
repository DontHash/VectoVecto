"""
test_app_document.py — Gradio document tab handler (in-process).
"""
from __future__ import annotations

import os
import sys

import cv2
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402


def test_create_app_builds():
    import app
    demo = app.create_app()
    assert demo is not None


@pytest.mark.skipif(not __import__("document_ocr").available_backends(),
                    reason="no OCR backend available")
def test_process_document_returns_outputs():
    import app
    from degradation_document import degrade_page

    page, _gt = doc_data.render_synthetic_invoice(seed=95, dpi=150)
    degraded = degrade_page(page, seed=9501, level="mild")
    rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

    slider, overlay, transcript, pdf, txt, status = app.process_document(
        rgb, None, "auto", False, True, True, True)

    assert slider is not None and len(slider) == 2
    assert slider[0].shape == rgb.shape and slider[1].shape == rgb.shape
    assert overlay is not None and overlay.shape == rgb.shape
    assert pdf and os.path.exists(pdf)
    assert txt and os.path.exists(txt)
    assert "Restore complete" in status
    assert "INVOICE" in transcript.upper() or "no low-confidence" in transcript.lower()


def test_process_document_all_pages_writes_combined(monkeypatch, tmp_path):
    import numpy as np

    import app
    import document_pipeline
    from document_ocr import OCRResult, Token

    class _FakeResult:
        def __init__(self, i):
            self.display_bgr = np.full((60, 40, 3), 255, np.uint8)
            self.ocr = OCRResult(text=f"PAGE {i} TEXT",
                                 tokens=[Token(text=f"PAGE{i}", conf=99,
                                               bbox=(2, 2, 36, 18))],
                                 backend="fake")
            self.meta = {"seconds": 0.1, "backend": "fake",
                         "primary_stream": "raw", "skew_angle": 0.0,
                         "resized": False}
            self.outputs = {}

        @property
        def status_line(self):
            return "fake"

    def fake_pages(path, dpi=200):
        for i in range(3):
            yield i, np.full((60, 40, 3), 255, np.uint8), f"gt{i}"

    monkeypatch.setattr(doc_data, "pdf_to_pages", fake_pages)
    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        lambda img, **kw: _FakeResult(int(kw["stem"][-1])))
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))

    pdf_in = tmp_path / "in.pdf"
    pdf_in.write_bytes(b"%PDF-1.4")
    _slider, _ov, _tr, pdf, txt, status = app.process_document(
        None, str(pdf_in), "auto", False, False, False, False, None, True)

    assert pdf and os.path.exists(pdf)
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    assert len(doc) == 3
    for i in range(3):
        assert f"PAGE{i}" in doc[i].get_textpage().get_text_range()
    assert txt and os.path.exists(txt)
    assert "3 processed" in status


def test_process_document_no_input():
    import app
    slider, overlay, transcript, pdf, txt, status = app.process_document(
        None, None, "auto", False, True, True, True)
    assert slider is None and "Upload" in status


def test_process_document_threads_lang(monkeypatch):
    import app
    import numpy as np
    import document_pipeline

    captured = {}

    def fake_pipeline(img, **kw):
        captured.update(kw)
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(document_pipeline, "run_document_pipeline", fake_pipeline)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    _slider, _ov, _tr, _pdf, _txt, status = app.process_document(
        img, None, "auto", False, True, True, True, "ne")
    assert captured.get("lang") == "ne"
    assert "failed" in status


@pytest.mark.skipif(not __import__("document_ocr").available_backends(),
                    reason="no OCR backend available")
def test_process_document_accepts_filepath_with_exif(tmp_path):
    import app
    from PIL import Image

    page, _gt = doc_data.render_synthetic_invoice(seed=96, dpi=150)
    stored = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)  # content 90 CW
    rgb = cv2.cvtColor(stored, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[274] = 8  # stored is 90 CW, display must rotate 90 CCW -> upright page
    path = tmp_path / "photo.jpg"
    im.save(path, exif=exif)

    slider, _overlay, _tr, _pdf, _txt, status = app.process_document(
        str(path), None, "auto", False, False, False, False)
    assert slider is not None, status
    assert slider[0].shape[:2] == page.shape[:2], \
        "the handler input must be EXIF-corrected before the pipeline"
