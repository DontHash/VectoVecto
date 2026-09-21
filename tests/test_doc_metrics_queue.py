"""
test_doc_metrics_queue.py — ranked review-queue quality over all tokens (W0.1).

`queue_stats` measures how many *token* errors (not just digit tokens) the
ranked queue surfaces in its top K. This is the metric the W0 improvement
waves gate on: flags must move real errors up, not just add noise.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_metrics import queue_stats  # noqa: E402
from document_ocr import Token  # noqa: E402


def _tok(text, conf=99, bbox=(0, 0, 10, 10), flags=None):
    return Token(text=text, conf=conf, bbox=bbox, flags=list(flags or []))


def test_queue_stats_counts_all_token_errors():
    gt = "नेपालको संविधान धारा"
    toks = [_tok("नेपालको"), _tok("संविधानन्", flags=["low_conf"]),
            _tok("धारा"), _tok("गलत", flags=["low_conf"])]
    s = queue_stats(toks, gt, top_k=(2,))
    assert s["errors"] == 2
    assert s["hits@2"] == 2
    assert s["recall@2"] == 1.0
    assert s["precision@2"] == 1.0


def test_queue_stats_ranks_by_risk_not_order():
    gt = "एक दुई"
    toks = [_tok("गलत", flags=["low_conf"]),
            _tok("भुल", flags=["digit_conflict"]), _tok("एक")]
    s = queue_stats(toks, gt, top_k=(1,))
    assert s["errors"] == 2
    assert s["recall@1"] == 0.5, "top-1 must be the riskiest error"


def test_queue_stats_page_without_errors():
    s = queue_stats([_tok("एक"), _tok("दुई")], "एक दुई", top_k=(5,))
    assert s["errors"] == 0
    assert s["recall@5"] is None
    assert s["precision@5"] is None


def test_queue_stats_punctuation_insensitive():
    gt = "धारा १,"
    toks = [_tok("धारा"), _tok("१")]
    s = queue_stats(toks, gt, top_k=(5,))
    assert s["errors"] == 0
