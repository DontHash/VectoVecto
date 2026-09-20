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
