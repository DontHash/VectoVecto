"""
test_document_orientation.py — EXIF load, OSD parsing, size gate, suspect flag.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import Token, available_backends  # noqa: E402
from document_orientation import (detect_rotation, load_image_bgr,  # noqa: E402
                                  orientation_suspect, parse_osd, rotate_bgr)

TESS_OK = "tesseract" in available_backends()


def test_parse_osd():
    sample = ("Page number: 0\n"
              "Orientation in degrees: 90\n"
              "Rotate: 270\n"
              "Orientation confidence: 6.44\n"
              "Script: Latin\n")
    parsed = parse_osd(sample)
    assert parsed == {"orientation": 90, "rotate": 270, "confidence": 6.44,
                      "script": "Latin"}
    assert parse_osd("no osd fields here") is None
    fallback = parse_osd("Orientation in degrees: 90\nOrientation confidence: 5.0\n")
    assert fallback["rotate"] == 270, "without Rotate, correct = 360 - orientation"


def test_rotate_bgr_shapes():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    assert rotate_bgr(img, 0).shape[:2] == (100, 300)
    assert rotate_bgr(img, 90).shape[:2] == (300, 100)
    assert rotate_bgr(img, 180).shape[:2] == (100, 300)
    assert rotate_bgr(img, 270).shape[:2] == (300, 100)


def test_detect_rotation_skips_small_pages():
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    info = detect_rotation(img)
    assert info.angle == 0 and info.source == "skipped-small"


def test_load_image_bgr_applies_exif(tmp_path):
    from PIL import Image

    arr = np.zeros((40, 80, 3), dtype=np.uint8)
    arr[:, :40] = 255
    im = Image.fromarray(arr)
    exif = Image.Exif()
    exif[274] = 6  # orientation: rotate 90 CW for display
    jpg = tmp_path / "exif.jpg"
    im.save(jpg, exif=exif)
    out = load_image_bgr(str(jpg))
    assert out.shape[:2] == (80, 40), "EXIF orientation must be applied at load"

    png = tmp_path / "plain.png"
    im.save(png)
    out2 = load_image_bgr(str(png))
    assert out2.shape[:2] == (40, 80)


def test_orientation_suspect_geometry():
    vertical = [Token("x", 99, (0, i * 50, 20, i * 50 + 45), "line")
                for i in range(6)]
    horizontal = [Token("x", 99, (i * 50, 0, i * 50 + 45, 20), "line")
                  for i in range(6)]
    assert orientation_suspect(vertical) is True
    assert orientation_suspect(horizontal) is False
    assert orientation_suspect([]) is False


@pytest.mark.skipif(not TESS_OK, reason="tesseract unavailable")
def test_osd_detects_rotated_synthetic_page():
    page, _gt = doc_data.render_synthetic_invoice(seed=311, dpi=150)
    assert detect_rotation(page).angle == 0
    assert detect_rotation(cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)).angle == 270
    assert detect_rotation(cv2.rotate(page, cv2.ROTATE_90_COUNTERCLOCKWISE)).angle == 90
    # 180 is refused by policy (destructive false positives): suspect only
    flipped = detect_rotation(cv2.rotate(page, cv2.ROTATE_180))
    assert flipped.angle == 0 and flipped.source == "osd-180"


@pytest.mark.skipif(not TESS_OK or "rapidocr" not in available_backends(),
                    reason="tesseract/rapidocr unavailable")
def test_pipeline_auto_rotates_sideways_page():
    from document_pipeline import run_document_pipeline

    page, gt = doc_data.render_synthetic_invoice(seed=312, dpi=150)
    sideways = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
    res = run_document_pipeline(sideways, backend="rapidocr")
    assert res.meta["auto_rotate"]["angle"] == 270
    assert res.meta["auto_rotate"]["source"] == "osd"
    assert doc_metrics.cer(gt, res.ocr.text) < 0.30, \
        "auto-rotated OCR must be near upright quality (garbage is ~0.8)"

    off = run_document_pipeline(sideways, backend="rapidocr", auto_rotate=False)
    assert off.meta["auto_rotate"]["angle"] == 0
    assert doc_metrics.cer(gt, off.ocr.text) > 0.5, \
        "without rotation the OCR is garbage - proof the gate matters"


@pytest.mark.skipif(not TESS_OK or "rapidocr" not in available_backends(),
                    reason="tesseract/rapidocr unavailable")
def test_pipeline_never_blindly_flips_180_but_flags_it():
    from document_pipeline import run_document_pipeline

    page, _gt = doc_data.render_synthetic_invoice(seed=313, dpi=150)
    flipped_img = cv2.rotate(page, cv2.ROTATE_180)
    res = run_document_pipeline(flipped_img, backend="rapidocr")
    assert res.meta["auto_rotate"]["angle"] == 0, "no blind 180 flips"
    assert res.meta["auto_rotate"]["source"] == "osd-180"
    assert res.meta["orientation_suspect"] is True
    assert "orientation?" in res.status_line
