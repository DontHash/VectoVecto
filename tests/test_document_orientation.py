"""
test_document_orientation.py — evidence-first orientation (votes + geometry + probe).

Measured separation this file guards (see document_orientation docstring):
  * cls 180-vote fraction: flipped pages 0.73-1.00, upright <=0.40
  * vertical token fraction: sideways >=0.91, upright <0.6 (0/30 SROIE)
  * a 90cw probe pass resolves axis (verticality drops) and direction (votes)
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
from document_orientation import (RotationInfo, detect_rotation,  # noqa: E402
                                  infer_angle, load_image_bgr,
                                  orientation_suspect, parse_osd, rotate_bgr,
                                  vertical_fraction)

RAPID_OK = "rapidocr" in available_backends()
TESS_OK = "tesseract" in available_backends()


def _flat(n=6):
    return [Token("x", 99, (0, i * 30, 120, i * 30 + 20), "line") for i in range(n)]


def _tall(n=6):
    return [Token("x", 99, (i * 30, 0, i * 30 + 20, 120), "line") for i in range(n)]


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


def test_vertical_fraction_and_suspect():
    assert vertical_fraction(_tall()) == 1.0
    assert vertical_fraction(_flat()) == 0.0
    assert vertical_fraction(_flat(3)) == 0.0, "too few tokens -> no opinion"
    assert orientation_suspect(_tall()) is True
    assert orientation_suspect(_flat()) is False
    assert orientation_suspect([]) is False


def test_infer_angle_unit_paths():
    assert infer_angle(_flat(), (1.0, 1.0, 6)).angle == 180
    assert infer_angle(_flat(), (0.73, 0.58, 6)).source == "votes-180", "hard receipt"
    assert infer_angle(_flat(), (0.79, 0.25, 6)).source == "vote-ambiguous", \
        "high frac without high-confidence votes: flag, never flip"
    assert infer_angle(_flat(), (0.4, 0.2, 6)).source == "upright"
    assert infer_angle(_flat(), None).source == "upright"
    assert infer_angle(_tall(), (0.3, 0.1, 6)).source == "probe-needed"
    assert infer_angle(_flat(3), (1.0, 1.0, 3)).source == "upright", "n < MIN_TOKENS"

    off = infer_angle(_flat(), (1.0, 1.0, 6), auto_rotate=False)
    assert off.angle == 0 and off.raw_angle == 180 and off.source == "off"
    off2 = infer_angle(_tall(), (0.3, 0.1, 6), auto_rotate=False)
    assert off2.angle == 0 and off2.raw_angle == 90 and off2.source == "off"
    off3 = infer_angle(_flat(), (0.79, 0.25, 6), auto_rotate=False)
    assert off3.raw_angle == 180


def test_detect_rotation_osd_fallback_is_size_gated():
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


@pytest.mark.skipif(not RAPID_OK, reason="rapidocr unavailable")
def test_line_orientation_votes_separate_flip():
    from document_ocr import get_backend, ocr_page

    page, _gt = doc_data.render_synthetic_invoice(seed=341, dpi=150)
    be = get_backend("rapidocr")
    up = ocr_page(page, backend="rapidocr")
    fl = ocr_page(cv2.rotate(page, cv2.ROTATE_180), backend="rapidocr")
    v_up = be.line_orientation_votes(page, up)
    v_fl = be.line_orientation_votes(cv2.rotate(page, cv2.ROTATE_180), fl)
    assert v_fl is not None and v_up is not None
    assert v_fl[0] >= 0.6, f"flipped page must vote 180 (got {v_fl})"
    assert v_up[0] <= 0.5, f"upright page must not vote 180 (got {v_up})"


@pytest.mark.skipif(not RAPID_OK, reason="rapidocr unavailable")
@pytest.mark.parametrize("label,rot,expect", [
    ("up", None, 0),
    ("90cw", cv2.ROTATE_90_CLOCKWISE, 270),
    ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE, 90),
    ("180", cv2.ROTATE_180, 180),
])
def test_pipeline_resolves_every_orientation(label, rot, expect):
    from document_pipeline import run_document_pipeline

    page, gt = doc_data.render_synthetic_invoice(seed=342, dpi=150)
    img = page if rot is None else cv2.rotate(page, rot)
    res = run_document_pipeline(img, backend="rapidocr")
    assert res.meta["auto_rotate"]["angle"] == expect, label
    assert not res.meta["orientation_suspect"], label
    assert doc_metrics.cer(gt, res.ocr.text) < 0.05, \
        f"{label}: corrected page must OCR near-perfectly"
    assert res.meta["orientation_evidence"]["passes"] == (1 if expect == 0 else
                                                          2 if expect in (90, 180) else 3)


@pytest.mark.skipif(not RAPID_OK, reason="rapidocr unavailable")
def test_rotate_off_flags_instead_of_rotating():
    from document_pipeline import run_document_pipeline

    page, _gt = doc_data.render_synthetic_invoice(seed=343, dpi=150)
    sideways = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
    res = run_document_pipeline(sideways, backend="rapidocr", auto_rotate=False)
    assert res.meta["auto_rotate"]["angle"] == 0
    assert res.meta["auto_rotate"]["raw_angle"] == 90
    assert res.meta["orientation_suspect"] is True
    assert "orientation?" in res.status_line

    flipped = cv2.rotate(page, cv2.ROTATE_180)
    res2 = run_document_pipeline(flipped, backend="rapidocr", auto_rotate=False)
    assert res2.meta["auto_rotate"]["raw_angle"] == 180
    assert res2.meta["orientation_suspect"] is True


@pytest.mark.skipif(not TESS_OK or not RAPID_OK, reason="tesseract/rapidocr unavailable")
def test_osd_fallback_still_available_for_tesseract_backend():
    page, _gt = doc_data.render_synthetic_invoice(seed=311, dpi=150)
    assert detect_rotation(page).angle == 0
    assert detect_rotation(cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)).angle == 270
    flipped = detect_rotation(cv2.rotate(page, cv2.ROTATE_180))
    assert flipped.angle == 0 and flipped.source == "osd-180", \
        "OSD fallback never blind-flips: it flags"
