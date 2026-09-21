"""
test_document_verifier.py — cross-model digit verification contract.

Flag-only: a disagreeing second read raises `cross_model_conflict` and stores
the alternative in `alt_text`; text is never modified. Heavy backends are not
touched here (fake verifier callables).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from document_ocr import (RISK_WEIGHTS, Token, apply_digit_verifier,  # noqa: E402
                          available_backends)
from document_verifier import get_digit_verifier  # noqa: E402


def _tok(text, conf=90, bbox=(10, 10, 120, 40)):
    return Token(text=text, conf=conf, bbox=bbox)


def test_verifier_flags_disagreement_and_keeps_text():
    img = np.zeros((80, 200, 3), dtype=np.uint8)
    tok = _tok("१२०.५०")
    calls = {}

    def fake(crops):
        calls["n"] = len(crops)
        return ["120.00"]  # different digits

    conflicts = apply_digit_verifier([tok], img, fake, only_flagged=False)
    assert conflicts == 1
    assert "cross_model_conflict" in tok.flags
    assert tok.alt_text == "120.00"
    assert tok.text == "१२०.५०", "text must never change"
    assert calls["n"] == 1


def test_verifier_agreement_no_flag():
    img = np.zeros((80, 200, 3), dtype=np.uint8)
    tok = _tok("120.50")
    conflicts = apply_digit_verifier([tok], img, lambda crops: ["120.50"],
                                     only_flagged=False)
    assert conflicts == 0
    assert tok.flags == []
    assert tok.alt_text is None


def test_verifier_single_crop_callable_supported():
    img = np.zeros((80, 200, 3), dtype=np.uint8)
    tok = _tok("१२३")
    conflicts = apply_digit_verifier([tok], img, lambda crop: "456",
                                     only_flagged=False)
    assert conflicts == 1


def test_verifier_only_flagged_and_capped():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    flagged = _tok("११", bbox=(0, 0, 100, 30), conf=70)
    flagged.flags.append("digit_uncertain")
    unflagged = _tok("२२", bbox=(0, 50, 100, 80))
    extra = [_tok("३३", bbox=(0, 100, 100, 130)) for _ in range(3)]
    for t in extra:
        t.flags.append("low_conf")

    seen = []

    def fake(crops):
        seen.append(len(crops))
        return ["99"] * len(crops)

    conflicts = apply_digit_verifier([flagged, unflagged] + extra, img, fake,
                                     max_tokens=2)
    assert seen == [2], "only flagged suspects, capped at max_tokens"
    assert conflicts == 2
    assert unflagged.flags == []


def test_verifier_weight_in_queue():
    assert RISK_WEIGHTS["cross_model_conflict"] >= RISK_WEIGHTS["digit_uncertain"]


@pytest.mark.skipif("rapidocr" not in available_backends(),
                    reason="rapidocr unavailable")
def test_pipeline_verifier_scope_all_covers_unflagged_digit_tokens():
    import doc_data
    from document_pipeline import run_document_pipeline

    page, _gt = doc_data.render_synthetic_invoice(seed=301, dpi=150)

    def fake(crops):
        return ["9"] * len(crops)  # always disagrees

    flagged = run_document_pipeline(page, backend="rapidocr",
                                    digit_verifier=fake)
    allscope = run_document_pipeline(page, backend="rapidocr",
                                     digit_verifier=fake, verifier_scope="all")
    f = flagged.ocr.meta.get("digit_verifier", {}).get("conflicts", 0)
    a = allscope.ocr.meta.get("digit_verifier", {}).get("conflicts", 0)
    assert allscope.ocr.meta["digit_verifier"]["scope"] == "all"
    assert a >= f
    assert a > 0


def test_get_digit_verifier_factory():
    assert get_digit_verifier(None) is None
    assert get_digit_verifier("off") is None
    v = get_digit_verifier("bodhan")
    assert v is not None and v.name == "bodhan"
    try:
        get_digit_verifier("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
