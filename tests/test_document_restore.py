"""
test_document_restore.py — unit tests for the classical restore core.

Covers the two things that silently destroyed OCR when wrong:
  * deskew sign/confidence (a shadowed page must NOT get rotated)
  * illumination flattening (gradient must be removed, ink kept)
Plus the numpy-in/numpy-out contract and stream selection.
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
from document_restore import (  # noqa: E402
    build_ocr_stream, estimate_skew, flatten_illumination, restore_document, rotate,
)


def _projection_variance(img_bgr: np.ndarray, angle: float) -> float:
    g = cv2.cvtColor(rotate(img_bgr, angle), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).astype(np.float32).sum(axis=1).var())


def test_q0_deskew_returns_correction_angle():
    page, _ = doc_data.render_synthetic_invoice(seed=2, dpi=150)
    tilted = rotate(page, 5.0)
    gray = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)
    est = estimate_skew(gray)
    assert abs(est - (-5.0)) < 0.75, f"expected ~-5 correction, got {est}"
    var_fixed = _projection_variance(tilted, est)
    var_broken = _projection_variance(tilted, 0.0)
    assert var_fixed > 2.0 * var_broken, "correction must sharpen text rows"


def test_deskew_guard_ignores_shadow_gradient():
    page, _ = doc_data.render_synthetic_invoice(seed=4, dpi=150)
    h, w = page.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = (0.55 * xx / w + 0.2 * yy / h)
    shadowed = np.clip(page.astype(np.float32) * (1.0 - gradient[:, :, None]), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(shadowed, cv2.COLOR_BGR2GRAY)
    flat, _ = flatten_illumination(gray)
    est_raw = estimate_skew(gray)
    est_flat = estimate_skew(flat)
    assert abs(est_flat) < 2.0, f"flattened page should not be deskewed (got {est_flat})"
    assert abs(est_raw) < 4.0 or abs(est_flat) <= abs(est_raw)


def test_illumination_flatten_removes_gradient():
    page, _ = doc_data.render_synthetic_invoice(seed=6, dpi=150)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gain = 1.0 - 0.45 * xx / w
    shaded = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    flat, _ = flatten_illumination(shaded)
    row_spread_before = float(np.percentile(shaded, 90) - np.percentile(shaded, 10))
    row_spread_after = float(np.percentile(flat, 90) - np.percentile(flat, 10))
    assert row_spread_after < row_spread_before
    col_means_before = shaded.mean(axis=0)
    col_means_after = flat.mean(axis=0)
    assert (col_means_after.std() < col_means_before.std()), "column lighting bias must shrink"


def test_restore_contract():
    page, _ = doc_data.render_synthetic_invoice(seed=8, dpi=150)
    shaded = page.copy()
    shaded = np.clip(shaded.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    out = restore_document(shaded)
    assert set(out.keys()) == {"display_bgr", "ocr_bgr", "debug"}
    for key in ("display_bgr", "ocr_bgr"):
        assert out[key].dtype == np.uint8
        assert out[key].ndim == 3 and out[key].shape[2] == 3
    dbg = out["debug"]
    for key in ("skew_angle", "scale_factor", "illum", "binary", "ocr_stream", "seconds"):
        assert key in dbg


def test_restore_max_side_downscales():
    page, _ = doc_data.render_synthetic_invoice(seed=9, dpi=300)
    out = restore_document(page, max_side=600)
    assert max(out["display_bgr"].shape[:2]) <= 600
    assert out["debug"]["scale_factor"] < 1.0


def test_ocr_streams():
    gray = np.full((200, 200), 200, dtype=np.uint8)
    cv2.putText(gray, "1200.00", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 30, 3)
    for mode in ("clahe", "sauvola", "gray"):
        stream = build_ocr_stream(gray, mode)
        assert stream.shape == (200, 200, 3) and stream.dtype == np.uint8
    with pytest.raises(ValueError):
        build_ocr_stream(gray, "nope")
