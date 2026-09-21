"""
test_error_taxonomy.py — token error classification (R1).

`classify_token_errors` turns a GT line vs OCR line diff into counts:
digit / matra / consonant / order / segmentation / missing / invented / other.
Scored on heiDATA (human-corrected ALTO), so the taxonomy itself is trustworthy.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from eval_error_taxonomy import classify_token_errors  # noqa: E402


def test_identical_lines_have_no_errors():
    c = classify_token_errors("नेपालको संविधान", "नेपालको संविधान")
    assert c["digit"] == 0 and c["matra"] == 0 and c["consonant"] == 0
    assert c["missing"] == 0 and c["invented"] == 0
    assert c["gt_tokens"] == 2


def test_digit_error():
    c = classify_token_errors("धारा १२ को", "धारा १३ को")
    assert c["digit"] == 1
    assert c["matra"] == 0 and c["consonant"] == 0


def test_matra_error():
    c = classify_token_errors("कि त", "कु त")
    assert c["matra"] == 1


def test_consonant_error():
    c = classify_token_errors("काम", "गाम")
    assert c["consonant"] == 1


def test_order_error_same_characters():
    c = classify_token_errors("कि", "िक")
    assert c["order"] == 1


def test_segmentation_error():
    c = classify_token_errors("नेपालको", "नेपाल को")
    assert c["segmentation"] == 1


def test_missing_and_invented():
    c = classify_token_errors("नेपालको संविधान धारा", "संविधान अनुच्छेद")
    assert c["missing"] == 2
    assert c["invented"] == 1


def test_mixed_line_counts_everything():
    c = classify_token_errors("धारा १२ को काम", "धारा १३ को काम नयाँ")
    assert c["digit"] == 1
    assert c["invented"] == 1
    assert c["gt_tokens"] == 4
