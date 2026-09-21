"""
test_document_flags.py — Devanagari flag signals + confidence calibration.

Signals are tuned on data/doc_eval/heidata_dev (Appendix M) and must never
alter text. All measurement in the commit messages/appendices; these tests
pin the mechanics.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from calibration import apply_isotonic, load_calibration  # noqa: E402
from document_ocr import (RISK_WEIGHTS, Token, flag_tokens,  # noqa: E402
                          review_queue, token_risk)


def _tok(text, conf, bbox=(0, 0, 10, 10), flags=None):
    return Token(text=text, conf=conf, bbox=bbox, flags=list(flags or []))


def test_script_mismatch_only_on_devanagari_pages():
    toks = [_tok("EKLL", 90), _tok("नेपाल", 90)]
    flag_tokens(toks, conf_threshold=60, devanagari=True)
    assert "script_mismatch" in toks[0].flags
    assert "script_mismatch" not in toks[1].flags

    toks2 = [_tok("EKLL", 90), _tok("hello", 90)]
    flag_tokens(toks2, conf_threshold=60, devanagari=False)
    assert not any("script_mismatch" in t.flags for t in toks2)


def test_devanagari_digit_bar_is_higher():
    tok = _tok("१२०", 85)
    flag_tokens([tok], conf_threshold=60, devanagari=True)
    assert "digit_uncertain" in tok.flags

    tok2 = _tok("120", 85)
    flag_tokens([tok2], conf_threshold=60, devanagari=False)
    assert "digit_uncertain" not in tok2.flags


def test_flags_never_change_text():
    toks = [_tok("एक १२ गलत", 50), _tok("ABC", 99)]
    before = [(t.text, t.conf) for t in toks]
    flag_tokens(toks, conf_threshold=60, devanagari=True)
    assert [(t.text, t.conf) for t in toks] == before


def test_invalid_sequence_flag_on_broken_devanagari():
    toks = [_tok("िक", 99), _tok("नेपाल", 99), _tok("क्ा", 99)]
    flag_tokens(toks, conf_threshold=60, devanagari=True)
    assert "invalid_sequence" in toks[0].flags, "reordered matra is a misread"
    assert "invalid_sequence" not in toks[1].flags
    assert "invalid_sequence" in toks[2].flags


def test_invalid_sequence_ignores_latin_and_valid_marks():
    toks = [_tok("Section 12", 99), _tok("तपाईंले", 99), _tok("हाँ", 99)]
    flag_tokens(toks, conf_threshold=60, devanagari=True)
    assert not any("invalid_sequence" in t.flags for t in toks)


def test_invalid_sequence_is_queued_below_digit_signals():
    bad = _tok("िक", 99, flags=["invalid_sequence"])
    digit = _tok("१२", 99, flags=["digit_conflict"])
    plain = _tok("नेपाल", 99)
    assert RISK_WEIGHTS["invalid_sequence"] >= RISK_WEIGHTS["low_conf"]
    assert RISK_WEIGHTS["digit_conflict"] > RISK_WEIGHTS["invalid_sequence"]
    queue = review_queue([plain, bad, digit])
    assert queue[0].text == "१२", "digit signals must keep the top of the queue"
    assert queue[1].text == "िक"
    assert token_risk(plain) == 0.0


def test_calibration_sets_cal_conf_and_preserves_ranking_order():
    toks = [_tok("क", 99), _tok("ख", 85), _tok("ग", 60)]
    cal = load_calibration()
    assert cal is not None, "calibration/rapidocr_devanagari_v1.json must ship"
    flag_tokens(toks, conf_threshold=60, devanagari=True, calibration=cal)
    cals = [t.cal_conf for t in toks]
    assert all(c is not None and 0 <= c <= 100 for c in cals)
    assert cals[0] >= cals[1] >= cals[2], "isotonic map is monotone"


def test_apply_isotonic_monotone_and_bounded():
    mapping = {"x": [0.5, 0.8, 0.9], "y": [0.1, 0.3, 0.55]}
    vals = [apply_isotonic(c, mapping) for c in (30, 55, 70, 85, 95, 100)]
    assert vals == sorted(vals)
    assert all(0 <= v <= 100 for v in vals)
    assert apply_isotonic(-5, mapping) == pytest.approx(10.0)
    assert apply_isotonic(150, mapping) == pytest.approx(55.0)


def test_script_mismatch_outranks_low_conf_and_joins_queue():
    mist = _tok("EKLL", 99, flags=["script_mismatch"])
    low = _tok("abc", 40, flags=["low_conf"])
    assert token_risk(mist) > token_risk(low)
    assert RISK_WEIGHTS["script_mismatch"] > RISK_WEIGHTS["low_conf"]
    queue = review_queue([low, mist])
    assert queue[0].text == "EKLL"


def test_calibration_file_metadata():
    cal = load_calibration()
    assert cal["domain"] == "heidata_dev"
    assert cal["ece_isotonic"] < 0.2
    assert cal["temperature_evidence"]["direction"] == "normal"
    assert os.path.exists(os.path.join(BASE_DIR, "calibration",
                                       "rapidocr_devanagari_v1.json"))
