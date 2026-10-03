"""
test_document_reconcile.py — track B: page-level constraint reconciliation
(docs/HARNESS_PLAN.md §4).

Contracts under test: a candidate reading supported by the page (date-column
majority, prefix-group majority) raises `context_conflict` exactly once and
becomes a suggestion; ambiguity is never flagged; `text` and `alt_text` are
never touched; the pass is deterministic and idempotent.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document.ocr import Token  # noqa: E402
from veriscript.document.reconcile import reconcile_page  # noqa: E402


def _col(texts, x0=100, x1=300, step=40, **kw):
    return [Token(text=t, conf=80.0, bbox=(x0, 10 + step * i, x1,
                                           10 + step * i + 30),
                  backend="fake", **kw)
            for i, t in enumerate(texts)]


def test_date_column_majority_flags_candidate():
    toks = _col(["मिति २०८१-०४-२७", "मिति २०८१-०४-२८",
                 "मिति २०८१-०४-२९", "मिति २०८१-०४-३०",
                 "मिति २०८२-०४-२७"])
    deviating = toks[-1]
    deviating.repass_text = "मिति २०८१-०४-२७"  # the re-pass saw the majority year

    stats = reconcile_page(toks)

    assert stats["flags"] == 1 and stats["date_flags"] == 1
    assert deviating.flags == ["context_conflict"]
    assert deviating.text == "मिति २०८२-०४-२७", "text is never edited"
    assert deviating.alt_text is None
    assert deviating.suggestions == [{
        "text": "मिति २०८१-०४-२७", "source": "context",
        "why": "date column: 5 tokens share year/month"}]
    assert all("context_conflict" not in t.flags for t in toks[:-1])


def test_date_column_without_candidate_is_not_flagged():
    toks = _col(["मिति २०८१-०४-२७", "मिति २०८१-०४-२८",
                 "मिति २०८१-०४-२९", "मिति २०८१-०४-३०",
                 "मिति २०८२-०४-२७"])
    stats = reconcile_page(toks)

    assert stats["flags"] == 0
    assert stats["candidates"] == 0


def test_date_column_ambiguity_is_never_flagged():
    toks = _col(["मिति २०८१-०४-२७", "मिति २०८१-०४-२८",
                 "मिति २०८१-०४-२९", "मिति २०८१-०४-३०",
                 "मिति २०८२-०४-२७"])
    deviating = toks[-1]
    deviating.repass_text = "मिति २०८१-०६-२७"
    deviating.alt_text = "मिति २०८१-०७-२७"  # two candidates both match the year

    stats = reconcile_page(toks)

    assert stats["flags"] == 0, "ambiguity is not evidence"
    assert deviating.flags == []


def test_prefix_column_flags_first_group_candidate():
    toks = _col(["०७६-०२५३-१२३", "०७६-०२५३-१२४", "०७६-०२५३-१२५",
                 "०७६-०२५३-१२६", "०७५-०२५३-१२३"])
    deviating = toks[-1]
    deviating.repass_text = "०७६-०२५३-१२३"

    stats = reconcile_page(toks)

    assert stats["flags"] == 1 and stats["prefix_flags"] == 1
    assert deviating.flags == ["context_conflict"]
    assert deviating.text == "०७५-०२५३-१२३"
    assert deviating.suggestions[0]["text"] == "०७६-०२५३-१२३"
    assert deviating.suggestions[0]["source"] == "context"


def test_reconcile_is_idempotent():
    toks = _col(["मिति २०८१-०४-२७", "मिति २०८१-०४-२८",
                 "मिति २०८१-०४-२९", "मिति २०८१-०४-३०",
                 "मिति २०८२-०४-२७"])
    toks[-1].repass_text = "मिति २०८१-०४-२७"

    reconcile_page(toks)
    reconcile_page(toks)

    assert toks[-1].flags.count("context_conflict") == 1
    assert len([s for s in toks[-1].suggestions
                if s["text"] == "मिति २०८१-०४-२७"]) == 1
