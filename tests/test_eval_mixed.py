"""
test_eval_mixed.py — pre-registered mixed-router gates (P5).

Gates fixed before measuring: text CER <= plain +2% relative; non-text
regions >= the naive-restore baseline (PSNR/SSIM); no new invented tokens;
router cost <= +1 s/page.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from eval_mixed import gate_verdict, masked_psnr, masked_ssim  # noqa: E402


def test_masked_psnr_identical_is_infinite():
    a = np.full((20, 20, 3), 128, dtype=np.uint8)
    mask = np.full((20, 20), 255, dtype=np.uint8)
    assert masked_psnr(a, a, mask) == float("inf")


def test_masked_psnr_only_counts_masked_pixels():
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    b = np.zeros((20, 20, 3), dtype=np.uint8)
    b[0:5, 0:5] = 255  # big difference outside the mask
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10:15, 10:15] = 255
    assert masked_psnr(a, b, mask) == float("inf")
    mask[0:5, 0:5] = 255
    assert np.isfinite(masked_psnr(a, b, mask))


def test_masked_ssim_identical_is_one():
    a = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    assert abs(masked_ssim(a, a, mask) - 1.0) < 1e-6


def test_gate_verdict_all_pass():
    v = gate_verdict(cer_plain=0.10, cer_routed=0.101, psnr_routed=float("inf"),
                     psnr_baseline=20.0, invented=0, cost_s=0.05)
    assert v["pass"] is True


def test_gate_verdict_text_regression_fails():
    v = gate_verdict(cer_plain=0.10, cer_routed=0.11, psnr_routed=float("inf"),
                     psnr_baseline=20.0, invented=0, cost_s=0.05)
    assert v["text"] is False and v["pass"] is False


def test_gate_verdict_non_text_and_invented_and_cost_fail():
    v = gate_verdict(cer_plain=0.10, cer_routed=0.10, psnr_routed=18.0,
                     psnr_baseline=20.0, invented=3, cost_s=1.5)
    assert v["non_text"] is False and v["invented"] is False
    assert v["cost"] is False and v["pass"] is False
