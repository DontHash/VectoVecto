"""
test_doc_metrics_agreement.py — cross-reading bag agreement (N5 anchor).

`three_way_agreement` compares independent readings of the same page at the
token-set level: a token is agreed when every reading contains it. Pages where
text layer / RapidOCR / bodhan all agree are consensus-anchored; the rest go
to the human worksheet.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_metrics import three_way_agreement  # noqa: E402


def test_identical_readings_agree_fully():
    text = "नेपालको संविधान धारा १"
    a = three_way_agreement([text, text, text])
    assert a["agreement_rate"] == 1.0
    assert a["disagreements"] == []
    assert a["readings"] == 3


def test_one_disagreeing_reading_lowers_rate():
    a = three_way_agreement(["नेपालको संविधान", "नेपालको संविधान",
                             "नेपालको संविधानन्"])
    assert a["union_tokens"] == 3
    assert a["agreed_tokens"] == 1
    assert a["agreement_rate"] == round(1 / 3, 4)
    assert "संविधानन्" in a["disagreements"]
    assert "संविधान" in a["disagreements"]


def test_multiplicity_is_ignored():
    a = three_way_agreement(["धारा धारा १", "धारा १", "धारा धारा १"])
    assert a["agreement_rate"] == 1.0


def test_punctuation_only_differences_are_not_disagreements():
    a = three_way_agreement(["धारा १,", "धारा १", "धारा, १"])
    assert a["agreement_rate"] == 1.0


def test_empty_readings_do_not_crash():
    a = three_way_agreement(["", "", ""])
    assert a["union_tokens"] == 0
    assert a["agreement_rate"] == 0.0
