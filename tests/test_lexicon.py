"""
test_lexicon.py — Devanagari lexicon loader + word extraction (W-B).

The lexicon is optional at runtime (like calibration): absence must never
raise, and the flag mechanics are pinned here. Measurements live in
docs/EVALUATION.md; this file pins behaviour.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lexicon import (deva_words, is_deva_word, load_lexicon,  # noqa: E402
                     normalize_word, unknown_words)


def test_normalize_word_nfc_and_zwj_removed():
    assert normalize_word("\u0915\u094d\u200d\u0937") == "\u0915\u094d\u0937"
    assert normalize_word("  नेपाल  ") == "नेपाल"


def test_is_deva_word_rules():
    assert is_deva_word("नेपाल")
    assert not is_deva_word("नेपाल।"), "danda is not a word character"
    assert not is_deva_word("abc")
    assert not is_deva_word("क"), "below MIN_WORD_LEN"
    assert not is_deva_word("१२"), "digits are not lexicon words"


def test_deva_words_drops_digits_punctuation_and_short_tokens():
    assert deva_words("रु. १,२३४.५० कुल जम्मा") == ["रु", "कुल", "जम्मा"]
    assert deva_words("12.50") == []
    assert deva_words("क १२३") == []


def test_unknown_words_with_and_without_lexicon():
    lex = {"नेपाल", "कुल"}
    assert unknown_words("नेपाल कुल", lex) == []
    assert unknown_words("नेपाल गलत", lex) == ["गलत"]
    assert unknown_words("गलत", None) == []
    assert unknown_words("गलत", frozenset()) == []


def test_load_lexicon_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "lex.txt"
    path.write_text("# header\n\nनेपाल\nकुल\n# note\n", encoding="utf-8")
    lex = load_lexicon(str(path))
    assert lex is not None
    assert lex["count"] == 2
    assert "नेपाल" in lex["words"] and "कुल" in lex["words"]
    assert lex["source"] == "lex.txt"


def test_load_lexicon_missing_or_malformed_is_none(tmp_path):
    assert load_lexicon(str(tmp_path / "nope.txt")) is None
    bad = tmp_path / "bad.txt"
    bad.write_text("abc\n123\n", encoding="utf-8")
    assert load_lexicon(str(bad)) is None


def test_load_lexicon_env_override(tmp_path, monkeypatch):
    path = tmp_path / "env_lex.txt"
    path.write_text("नेपाल\n", encoding="utf-8")
    monkeypatch.setenv("VERISCRIPT_LEXICON", str(path))
    lex = load_lexicon()
    assert lex is not None and lex["count"] == 1


def test_load_lexicon_legacy_env_alias(tmp_path, monkeypatch):
    path = tmp_path / "env_lex2.txt"
    path.write_text("नेपाल\n", encoding="utf-8")
    monkeypatch.delenv("VERISCRIPT_LEXICON", raising=False)
    monkeypatch.setenv("VECTOVECTO_LEXICON", str(path))
    lex = load_lexicon()
    assert lex is not None and lex["count"] == 1
