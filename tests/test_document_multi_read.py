"""
test_document_multi_read.py — track A: multi-read consensus on flagged tokens
(docs/HARNESS_PLAN.md §3).

Contracts under test: every independent read is recorded with provenance;
`text`, `conf` and `alt_text` are never touched; a panel disagreement raises
`multi_read_conflict` exactly once and becomes a suggestion; an existing
`digit_conflict` is never duplicated and a recorded re-pass is reused instead
of re-read; suspects are capped and implausibly wide boxes are skipped; the
`ocr_page` wiring is strictly opt-in.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document import ocr as document_ocr  # noqa: E402
from veriscript.document.ocr import (OCRResult, Token, apply_multi_read,  # noqa: E402
                                     ocr_page)


def _page(w: int = 400, h: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def _tok(text, bbox=(10, 10, 110, 50), flags=("digit_uncertain",), conf=80.0):
    return Token(text=text, conf=conf, bbox=bbox, backend="fake", flags=list(flags))


class _Panel:
    """recognize_crop fake: a fixed reading per call, call counter."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, crop, lang=None):
        text = (self.texts[self.calls] if self.calls < len(self.texts)
                else self.texts[-1])
        self.calls += 1
        return text, 90.0


def test_multi_read_records_reads_and_never_edits_text():
    tok = _tok("२०८१")
    panel = _Panel(["२०८१"])  # every transform agrees
    stats = apply_multi_read([tok], _page(), panel, lang="ne")

    assert tok.text == "२०८१" and tok.conf == 80.0 and tok.alt_text is None
    assert [r["source"] for r in tok.reads] == [
        "panel:2x", "panel:rot+1.5", "panel:otsu"]
    assert all(r["conf"] == 90.0 for r in tok.reads)
    assert tok.flags == ["digit_uncertain"], "agreement adds no flag"
    assert tok.suggestions == []
    assert stats == {"checked": 1, "agree": 1, "split": 0, "unreadable": 0,
                     "conflicts": 0}


def test_multi_read_disagreement_flags_once_and_suggests():
    tok = _tok("२०८१")
    stats = apply_multi_read([tok], _page(), _Panel(["२०८०"]), lang="ne")

    assert tok.text == "२०८१", "the engine reading is never replaced"
    assert tok.flags == ["digit_uncertain", "multi_read_conflict"]
    assert stats["split"] == 1 and stats["conflicts"] == 1
    assert tok.suggestions and tok.suggestions[0]["text"] == "२०८०"
    assert tok.suggestions[0]["source"].startswith("panel:")
    assert tok.alt_text is None, "panel candidates never write alt_text"

    # a second run must not duplicate the flag or the suggestion
    apply_multi_read([tok], _page(), _Panel(["२०८०"]), lang="ne")
    assert tok.flags.count("multi_read_conflict") == 1
    assert len([s for s in tok.suggestions if s["text"] == "२०८०"]) == 1


def test_multi_read_respects_digit_conflict_and_reuses_repass():
    tok = _tok("२०८१", flags=("digit_conflict",))
    tok.repass_text, tok.repass_conf = "२०८०", 88.0
    panel = _Panel(["२०८०"])
    stats = apply_multi_read([tok], _page(), panel, lang="ne")

    assert tok.flags == ["digit_conflict"], "no duplicate disagreement flag"
    assert stats["conflicts"] == 0
    assert tok.reads[0]["source"] == "repass"
    assert panel.calls == 2, "repass read reused; only rot + otsu are new calls"
    assert tok.suggestions, "candidates are still recorded for the human"


def test_multi_read_caps_and_skips_wide_boxes():
    toks = [_tok("१२३", bbox=(10, 10 + 30 * i, 110, 40 + 30 * i))
            for i in range(15)]
    wide = _tok("१२३", bbox=(0, 0, 900, 50))  # aspect 18 > 8
    panel = _Panel(["१२३"])
    stats = apply_multi_read(toks + [wide], _page(1000, 600), panel, lang="ne")

    assert stats["checked"] == 12, "cap of 12 suspects per page"
    assert wide.reads == [], "implausibly wide boxes are skipped"
    assert panel.calls == 12 * 3


class _FakeBackend:
    name = "fake"

    def __init__(self, tok):
        self.tok = tok
        self.calls = 0

    def available(self):
        return True, ""

    def run(self, img, lang=None):
        return OCRResult(text=self.tok.text, tokens=[self.tok],
                         backend="fake", meta={})

    def recognize_crop(self, crop, lang=None):
        self.calls += 1
        return "२०८१", 90.0


def test_ocr_page_multi_read_wires_the_panel(monkeypatch):
    img = _page(300, 100)
    tok = Token(text="२०८१", conf=50.0, bbox=(20, 20, 200, 60),
                backend="fake")
    backend = _FakeBackend(tok)
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)

    res = ocr_page(img, backend="fake", lang="ne", multi_read="panel")
    assert res.meta["multi_read"]["checked"] == 1
    assert backend.calls == 3

    off = ocr_page(img, backend="fake", lang="ne")
    assert "multi_read" not in off.meta, "the panel is strictly opt-in"
