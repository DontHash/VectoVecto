"""
test_document_ocr_state.py — engine-state regression tests for document_ocr.

Locks down the RapidOCR 3.x state-leak bug: a recognition-only call must not
disable detection for subsequent full-page OCR (upstream update_params()
setattr()s use_det/use_cls on the engine without restoring them).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from document_ocr import available_backends, get_backend, ocr_page  # noqa: E402

pytestmark = pytest.mark.skipif("rapidocr" not in available_backends(),
                                reason="rapidocr unavailable")


def _page():
    img = np.full((160, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TOTAL 1200.00", (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def test_recognize_crop_preserves_full_page_engine_state():
    page = _page()
    before = ocr_page(page, backend="rapidocr", conf_threshold=-1.0)
    assert before.tokens, "baseline OCR must find text"

    backend = get_backend("rapidocr")
    x0, y0, x1, y1 = before.tokens[0].bbox
    crop = page[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2].copy()
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    text, _conf = backend.recognize_crop(big)
    assert text.strip(), "crop re-read should return text"

    after = ocr_page(page, backend="rapidocr", conf_threshold=-1.0)
    assert after.tokens, ("full-page OCR returned zero tokens after a rec-only "
                          "call - RapidOCR state leak is back")
    assert after.text == before.text
