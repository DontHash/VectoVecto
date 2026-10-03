"""
test_document_memory.py — track D2: local, text-only correction memory
(docs/HARNESS_PLAN.md §5).

Contracts under test: disabled by default; text-only (no bbox, run id or crop
references); exact and near (Hamming <= 1) matching under the same flags
signature; suggestions suppressed after net-negative feedback; accept/reject
recorded from the raw correction entries; idempotent review augmentation;
clear removes the store.
"""
from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document import memory  # noqa: E402


def _enable(monkeypatch, tmp_path) -> str:
    monkeypatch.setenv("VERISCRIPT_MEMORY", "1")
    path = str(tmp_path / "mem.jsonl")
    monkeypatch.setenv("VERISCRIPT_MEMORY_DIR", path)
    return path


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VERISCRIPT_MEMORY", raising=False)
    path = str(tmp_path / "mem.jsonl")
    written = memory.record_corrections(
        [{"index": 0, "corrected": "X", "action": "changed"}],
        {0: ("Y", ["low_conf"])}, path=path)

    assert written == 0 and not os.path.exists(path)
    assert memory.suggest("Y", ["low_conf"], path=path) is None
    assert memory.augment_review([{"text": "Y", "flags": ["low_conf"]}],
                                 path=path) == 0


def test_exact_and_near_match(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "मिति २०८१-०४-३२", "action": "changed"}],
        {0: ("मिति २०८१-०४-३९", ["digit_conflict"])}, path=path)

    exact = memory.suggest("मिति २०८१-०४-३९", ["digit_conflict"], path=path)
    assert exact is not None and exact["text"] == "मिति २०८१-०४-३२"
    assert exact["source"] == "memory" and exact["why"]

    near = memory.suggest("मिति २०८१-०४-३८", ["digit_conflict"], path=path)
    assert near is not None and near["text"] == "मिति २०८१-०४-३२", \
        "Hamming distance 1 must match"

    assert memory.suggest("भिन्न २०८१-०४-३८", ["digit_conflict"], path=path) is None, \
        "a near match must share the non-digit skeleton"

    assert memory.suggest("अर्को २०८१-०४-३९", ["digit_conflict"], path=path) is None, \
        "same digits in a different reading must not match exactly"

    assert memory.suggest("मिति २०८१-०४-३९", ["low_conf"], path=path) is None, \
        "the flags signature is part of the key"


def test_feedback_suppresses_net_negative(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "१२३", "action": "changed"}],
        {0: ("१२४", ["digit_uncertain"])}, path=path)
    assert memory.suggest("१२४", ["digit_uncertain"], path=path) is not None

    for _ in range(2):  # the human corrected the memory suggestion away
        memory.record_corrections(
            [{"index": 0, "corrected": "१२५", "action": "changed",
              "suggested": "१२३"}],
            {0: ("१२४", ["digit_uncertain"])}, path=path)

    assert memory.suggest("१२४", ["digit_uncertain"], path=path) is None, \
        "net-negative feedback must suppress the record"


def test_accept_feedback_recorded(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "१२३", "action": "changed"}],
        {0: ("१२४", ["digit_uncertain"])}, path=path)
    memory.record_corrections(
        [{"index": 0, "corrected": "१२३", "action": "changed",
          "suggested": "१२३"}],
        {0: ("१२४", ["digit_uncertain"])}, path=path)

    suggestion = memory.suggest("१२४", ["digit_uncertain"], path=path)
    assert suggestion is not None and "accepted" in suggestion["why"]


def test_store_is_text_only(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "X", "action": "changed"}],
        {0: ("Y", ["low_conf"])}, path=path)

    line = json.loads(open(path, encoding="utf-8").readline())
    assert line["kind"] == "correction" and line["original"] == "Y"
    for forbidden in ("bbox", "run_id", "crop", "image", "page"):
        assert forbidden not in line, f"the store must stay text-only ({forbidden})"


def test_augment_review_is_idempotent(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "X2", "action": "changed"}],
        {0: ("X", ["low_conf"])}, path=path)
    review = [{"text": "X", "flags": ["low_conf"]}]

    assert memory.augment_review(review, path=path) == 1
    assert review[0]["suggestions"][0]["text"] == "X2"
    assert memory.augment_review(review, path=path) == 0


def test_value_memory_substitutes_repeated_run(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    # the human removed a spurious prefix from the case number
    memory.record_corrections(
        [{"index": 0, "corrected": "मुद्दा नं ०९६२ को", "action": "changed"}],
        {0: ("मुद्दा नं ०-०९६२ को", ["low_conf"])}, path=path)

    suggestion = memory.suggest_value("अर्को पृष्ठ ०-०९६२ बमोजिम", path=path)
    assert suggestion is not None and suggestion["source"] == "memory:value"
    assert suggestion["text"] == "अर्को पृष्ठ ०९६२ बमोजिम"
    assert "value" in suggestion["why"]


def test_value_memory_ignores_short_runs(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "दिन ३२", "action": "changed"}],
        {0: ("दिन ३१", ["low_conf"])}, path=path)

    assert memory.suggest_value("दिन ३१ मा", path=path) is None, \
        "runs below VALUE_MIN_DIGITS are not memory"


def test_value_feedback_suppresses(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "नं ०९६२", "action": "changed"}],
        {0: ("नं ०-०९६२", ["low_conf"])}, path=path)
    assert memory.suggest_value("नं ०-०९६२", path=path) is not None

    for _ in range(2):  # the human rejected the suggestion and kept the reading
        memory.record_corrections(
            [{"index": 0, "corrected": "नं ०-०९६२", "action": "confirmed",
              "suggested": "नं ०९६२"}],
            {0: ("नं ०-०९६२", ["low_conf"])}, path=path)

    assert memory.suggest_value("नं ०-०९६२", path=path) is None


def test_augment_review_falls_back_to_value(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "नं ०९६२", "action": "changed"}],
        {0: ("नं ०-०९६२", ["low_conf"])}, path=path)
    review = [{"text": "नं ०-०९६२ बमोजिम", "flags": ["low_conf"]}]

    assert memory.augment_review(review, path=path) == 1
    assert review[0]["suggestions"][0]["source"] == "memory:value"


def _fake_global(monkeypatch):
    monkeypatch.setattr(memory, "_global_words", lambda: frozenset({"राम"}))


def test_word_memory_suggests_confirmed_term(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    _fake_global(monkeypatch)
    memory.record_corrections(
        [{"index": 0, "corrected": "पौड्यालसमेत आए", "action": "changed"}],
        {0: ("पौड्यालसमे आए", ["low_conf"])}, path=path)

    suggestion = memory.suggest_word("पौड्यालसमे विरुद्ध", path=path)
    assert suggestion is not None and suggestion["source"] == "memory:word"
    assert suggestion["text"] == "पौड्यालसमेत विरुद्ध"
    assert suggestion["word"] == "पौड्यालसमेत"


def test_word_memory_skips_in_lexicon_words(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    _fake_global(monkeypatch)
    memory.record_corrections(
        [{"index": 0, "corrected": "राम आए", "action": "changed"}],
        {0: ("राम आए", ["low_conf"])}, path=path)

    assert memory.suggest_word("राम विरुद्ध", path=path) is None


def test_word_feedback_suppresses(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    _fake_global(monkeypatch)
    memory.record_corrections(
        [{"index": 0, "corrected": "पौड्यालसमेत आए", "action": "changed"}],
        {0: ("पौड्यालसमे आए", ["low_conf"])}, path=path)
    assert memory.suggest_word("पौड्यालसमे विरुद्ध", path=path) is not None

    for _ in range(2):  # the human rejected the term and kept the reading
        memory.record_corrections(
            [{"index": 0, "corrected": "पौड्यालसमे आए", "action": "confirmed",
              "suggested": "पौड्यालसमेत आए"}],
            {0: ("पौड्यालसमे आए", ["low_conf"])}, path=path)

    assert memory.suggest_word("पौड्यालसमे विरुद्ध", path=path) is None


def test_augment_review_falls_back_to_word(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    _fake_global(monkeypatch)
    memory.record_corrections(
        [{"index": 0, "corrected": "पौड्यालसमेत आए", "action": "changed"}],
        {0: ("पौड्यालसमे आए", ["low_conf"])}, path=path)
    review = [{"text": "पौड्यालसमे विरुद्ध", "flags": ["low_conf"]}]

    assert memory.augment_review(review, path=path) == 1
    assert review[0]["suggestions"][0]["source"] == "memory:word"


def test_clear_and_stats(tmp_path, monkeypatch):
    path = _enable(monkeypatch, tmp_path)
    memory.record_corrections(
        [{"index": 0, "corrected": "X", "action": "changed"}],
        {0: ("Y", ["low_conf"])}, path=path)

    assert memory.stats(path=path)["corrections"] == 1
    assert memory.clear(path=path) == 1
    assert not os.path.exists(path)
    assert memory.stats(path=path)["events"] == 0
