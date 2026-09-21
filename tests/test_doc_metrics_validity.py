"""
test_doc_metrics_validity.py — Devanagari GT integrity checker (N1).

`devanagari_validity` measures how many Devanagari tokens contain invalid
combining sequences (reordered matras, dangling viramas, orphan marks). It is
the harvest gate for text-layer ground truth: a clean text layer scores 0,
PDF-extraction damage shows up as a small but non-zero rate.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_metrics import devanagari_validity  # noqa: E402


def test_clean_devanagari_text_is_valid():
    text = "नेपालको संविधान २०७२ को धारा १, क्रियाशील"
    v = devanagari_validity(text)
    assert v["invalid_tokens"] == 0
    assert v["invalid_token_rate"] == 0.0
    assert v["devanagari_tokens"] == 7


def test_matra_then_anusvara_is_valid():
    v = devanagari_validity("हाँ किं")
    assert v["invalid_tokens"] == 0


def test_reordered_matra_is_invalid():
    v = devanagari_validity("िक देश")
    assert v["devanagari_tokens"] == 2
    assert v["invalid_tokens"] == 1
    assert v["examples"] == ["िक"]
    assert v["invalid_token_rate"] == 0.5


def test_dangling_virama_and_stacked_matras_are_invalid():
    v = devanagari_validity("क्ा क्ं क््")
    assert v["invalid_tokens"] == 3


def test_latin_and_digits_do_not_count_as_invalid():
    v = devanagari_validity("Section 12 of the Act")
    assert v["devanagari_tokens"] == 0
    assert v["invalid_tokens"] == 0
    assert v["invalid_token_rate"] == 0.0


def test_replacement_char_and_matra_after_digit():
    v = devanagari_validity("धारा\ufffd 12ा")
    assert v["invalid_tokens"] == 2


def test_empty_text():
    v = devanagari_validity("")
    assert v["tokens"] == 0
    assert v["invalid_token_rate"] == 0.0
    assert v["examples"] == []
