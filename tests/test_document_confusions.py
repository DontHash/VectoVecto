"""
test_document_confusions.py — measured digit-confusion chips
(docs/HARNESS_PLAN.md digit layer).

Contracts under test: only measured pairs generate; strongest pair first,
regardless of position; deterministic, capped, deduplicated; ASCII and
Devanagari both handled; chips are suggestion-only (text, conf and flags are
never touched) and only queued digit tokens receive them.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document.confusions import add_confusions, generate  # noqa: E402
from veriscript.document.ocr import Token  # noqa: E402


def test_generate_strongest_pair_first():
    chips = generate("मिति २०८१-०४-३९", limit=2)

    assert [c["text"] for c in chips] == ["मिति २०८१-०४-३१",
                                          "मिति २०८१-०४-३२"]
    assert all(c["source"] == "confusion" for c in chips)
    assert "measured confusion" in chips[0]["why"]


def test_generate_is_capped_deduped_and_conservative():
    assert len(generate("१९", limit=3)) == 3
    assert len({c["text"] for c in generate("१९", limit=3)}) == 3
    assert generate("५६७") == [], "digits with no measured partners generate nothing"
    assert generate("") == []


def test_generate_handles_ascii_digits():
    chips = generate("120", limit=2)

    assert [c["text"] for c in chips] == ["110", "126"]


def test_generate_prefers_page_supported_candidates():
    # the value २०८१-०४-३२ occurs elsewhere on the page -> it ranks first
    chips = generate("मिति २०८१-०४-३९", limit=2,
                     page_values={"20810432"})

    assert chips[0]["text"] == "मिति २०८१-०४-३२"
    assert chips[1]["text"] == "मिति २०८१-०४-३१"


def test_add_confusions_only_touches_queued_digit_tokens():
    queued = Token(text="२०८१-०४-३९", conf=80, bbox=(0, 0, 100, 30),
                   flags=["digit_uncertain"])
    unflagged = Token(text="२०८१-०४-३९", conf=99, bbox=(0, 40, 100, 70))
    touched = add_confusions([queued, unflagged])

    assert touched == 1
    assert queued.text == "२०८१-०४-३९" and queued.conf == 80
    assert queued.flags == ["digit_uncertain"], "suggestion-only: no new flags"
    assert unflagged.suggestions == []
    assert [s["text"] for s in queued.suggestions] == ["२०८१-०४-३१",
                                                       "२०८१-०४-३२"]


def test_add_confusions_caps_tokens_per_page():
    toks = [Token(text="१९", conf=50, bbox=(0, i * 30, 50, i * 30 + 20),
                  flags=["low_conf"]) for i in range(15)]

    touched = add_confusions(toks, max_tokens=3)

    assert touched == 3
    assert sum(1 for t in toks if t.suggestions) == 3
