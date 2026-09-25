"""
test_profile_textbook_scans.py — W-D unlabeled probe aggregation (no OCR).
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import profile_textbook_scans as pt  # noqa: E402


def _row(tokens=10, flagged=2, queue=1, mismatch=0, oov=1, digits=3,
         suspect=False, seconds=1.0, lexicon="nepali_lexicon_v1.txt:10"):
    return {"tokens": tokens, "flagged": flagged, "queue": queue,
            "script_mismatch": mismatch, "unknown_word": oov,
            "digit_tokens": digits, "orientation_suspect": suspect,
            "seconds": seconds, "lexicon": lexicon}


def test_aggregate_shares_and_median():
    rows = [_row(seconds=1.0), _row(seconds=3.0)]
    out = pt.aggregate(rows)
    assert out["pages"] == 2
    assert out["tokens"] == 20
    assert out["tokens_per_page"] == 10.0
    assert out["flagged_share"] == 0.2
    assert out["queue_share"] == 0.1
    assert out["unknown_word_share"] == 0.1
    assert out["digit_token_share"] == 0.3
    assert out["seconds_per_page_median"] == 2.0
    assert out["orientation_suspect_pages"] == 0
    assert out["lexicon"].startswith("nepali_lexicon")


def test_aggregate_empty_and_suspect_count():
    assert pt.aggregate([]) == {"pages": 0, "tokens": 0}
    out = pt.aggregate([_row(suspect=True), _row()])
    assert out["orientation_suspect_pages"] == 1


def test_aggregate_without_lexicon():
    out = pt.aggregate([_row(lexicon=None)])
    assert out["lexicon"] is None
    assert out["unknown_word_share"] == 0.1
