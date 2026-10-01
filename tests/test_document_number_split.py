"""
test_document_number_split.py — merged number-run splitting (Appendix AG).

The detector sometimes glues adjacent numbers (dates, case numbers) into one
box; the splitter cuts the box at physical separators (a wide clean gap or a
drawn table rule) and re-reads the segments. The contracts under test:

* a wide internal gap splits; tight spacing does not;
* uniformly wide digit spacing does not split (the median guard);
* a drawn rule is a boundary even when ink touches it from both sides;
* an empty segment read aborts and keeps the original token (text is never
  dropped, ink is never invented);
* non-digit or non-dominant tokens are untouched and cost no recognition.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document import ocr as document_ocr  # noqa: E402
from veriscript.document.ocr import (OCRResult, Token, _rule_continues,  # noqa: E402
                                     _segments_between, _split_boundaries,
                                     ocr_page, split_merged_number_tokens)


def _recognizer(texts):
    calls = []

    def fn(crop, lang=None):
        calls.append(crop.shape[:2])
        idx = len(calls) - 1
        return (texts[idx], 88.0) if idx < len(texts) else ("", 0.0)

    return fn, calls


def _put(img, text, x, y=110, scale=2.0):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)


def _ink_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 128)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _page(w=600, h=160):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_splits_two_numbers_at_a_wide_gap():
    img = _page()
    _put(img, "1234", 40)
    _put(img, "5678", 300)
    bbox = _ink_bbox(img)
    tok = Token(text="१२३४५६७८", conf=90.0, bbox=bbox, backend="fake")
    fn, calls = _recognizer(["१२३४", "५६७८"])

    out, stats = split_merged_number_tokens([tok], img, fn, lang="ne")

    assert stats["split"] == 1 and stats["segments"] == 2
    assert [t.text for t in out] == ["१२३४", "५६७८"]
    assert all(t.text_source == "split" for t in out)
    assert out[0].bbox[0] == bbox[0] and out[1].bbox[2] == bbox[2]
    assert out[0].bbox[2] <= out[1].bbox[0]
    assert len(calls) == 2


def test_tight_spacing_does_not_split():
    img = _page()
    _put(img, "12345678", 40)
    bbox = _ink_bbox(img)
    tok = Token(text="१२३४५६७८", conf=90.0, bbox=bbox, backend="fake")
    fn, calls = _recognizer(["x", "y"])

    out, stats = split_merged_number_tokens([tok], img, fn, lang="ne")

    assert stats["split"] == 0 and stats["aborted"] == 0
    assert out == [tok] and calls == []


def test_uniformly_wide_digit_spacing_does_not_split():
    img = _page(w=700)
    for i, ch in enumerate("12345678"):
        _put(img, ch, 40 + i * 70)
    bbox = _ink_bbox(img)
    tok = Token(text="१२३४५६७८", conf=90.0, bbox=bbox, backend="fake")
    fn, calls = _recognizer(["x", "y"])

    out, stats = split_merged_number_tokens([tok], img, fn, lang="ne")

    assert stats["split"] == 0
    assert out == [tok] and calls == []


def test_empty_segment_read_aborts_and_keeps_the_token():
    img = _page()
    _put(img, "1234", 40)
    _put(img, "5678", 300)
    bbox = _ink_bbox(img)
    tok = Token(text="१२३४५६७८", conf=90.0, bbox=bbox, backend="fake")
    fn, _calls = _recognizer(["१२३४", ""])

    out, stats = split_merged_number_tokens([tok], img, fn, lang="ne")

    assert out == [tok]
    assert stats["aborted"] == 1 and stats["split"] == 0


def test_non_candidate_tokens_are_untouched():
    img = _page()
    _put(img, "1234", 40)
    _put(img, "5678", 300)
    bbox = _ink_bbox(img)
    tok = Token(text="कुल ५", conf=90.0, bbox=bbox, backend="fake")
    fn, calls = _recognizer(["x", "y"])

    out, stats = split_merged_number_tokens([tok], img, fn, lang="ne")

    assert out == [tok] and calls == []
    assert stats["candidates"] == 0


def test_drawn_rule_is_a_boundary_even_without_a_wide_gap():
    # synthetic ink band: left mass | 2px gap | 3px full-height rule | 2px gap | right mass
    ink = np.zeros((40, 100), dtype=bool)
    ink[:, 0:30] = True
    ink[:, 32:35] = True
    ink[:, 37:100] = True

    boundaries = _split_boundaries(ink, 40)
    assert boundaries == [(32, 35)]
    segments = _segments_between(0, 100, boundaries)
    assert segments == [(0, 32), (35, 100)]


def test_rule_check_can_veto_a_glyph_stroke():
    ink = np.zeros((40, 100), dtype=bool)
    ink[:, 0:30] = True
    ink[:, 32:35] = True   # thin full-height run: stroke or rule?
    ink[:, 37:100] = True

    assert _split_boundaries(ink, 40,
                             rule_check=lambda a, b: False) == []
    assert _split_boundaries(ink, 40,
                             rule_check=lambda a, b: True) == [(32, 35)]


def test_rule_continues_requires_page_context():
    gray = np.full((200, 100), 255, dtype=np.uint8)
    gray[60:110, 40:44] = 0     # a stroke that stops at the token band edge
    assert not _rule_continues(gray, 40, 44, 60, 110, thr=128)

    gray[20:150, 40:44] = 0     # a rule that runs past the text row
    assert _rule_continues(gray, 40, 44, 60, 110, thr=128)


def test_wide_gap_ratio_boundary():
    ink = np.zeros((40, 140), dtype=bool)
    ink[:, 0:40] = True
    ink[:, 100:140] = True

    boundaries = _split_boundaries(ink, 40)
    assert boundaries == [(40, 100)]


class _FakeBackend:
    name = "fake"

    def __init__(self, tok, parts):
        self.tok = tok
        self.parts = list(parts)
        self.n = 0

    def run(self, img, lang=None):
        return OCRResult(text=self.tok.text, tokens=[self.tok],
                         backend="fake", meta={})

    def recognize_crop(self, crop, lang=None):
        text = self.parts[self.n] if self.n < len(self.parts) else ""
        self.n += 1
        return text, 88.0


def test_ocr_page_split_numbers_wires_the_splitter(monkeypatch):
    img = _page()
    _put(img, "1234", 40)
    _put(img, "5678", 300)
    bbox = _ink_bbox(img)
    tok = Token(text="१२३४५६७८", conf=90.0, bbox=bbox, backend="fake")
    backend = _FakeBackend(tok, ["१२३४", "५६७८"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)

    res = ocr_page(img, backend="fake", lang="ne", split_numbers=True)

    assert [t.text for t in res.tokens] == ["१२३४", "५६७८"]
    assert res.text == "१२३४\n५६७८"
    assert res.meta["number_split"]["split"] == 1


def test_ocr_page_split_off_keeps_backend_output(monkeypatch):
    img = _page()
    _put(img, "1234", 40)
    _put(img, "5678", 300)
    bbox = _ink_bbox(img)
    tok = Token(text="१२३४५६७८", conf=90.0, bbox=bbox, backend="fake")
    backend = _FakeBackend(tok, ["१२३४", "५६७८"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)

    res = ocr_page(img, backend="fake", lang="ne")

    assert [t.text for t in res.tokens] == ["१२३४५६७८"]
    assert "number_split" not in res.meta
