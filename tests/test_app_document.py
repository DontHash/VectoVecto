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


def test_process_document_no_input():
    import app
    slider, overlay, transcript, pdf, txt, status = app.process_document(
        None, None, "auto", False, True, True, True)
    assert slider is None and "Upload" in status


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
