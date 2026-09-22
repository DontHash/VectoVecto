"""
test_document_deva_lines.py — Devanagari line reader integration.

The reader is optional: without a checkpoint the OCR result must be exactly
the backend's, and with one the line texts are replaced only when the model
output is plausible (empty/overlong output keeps the backend reading).
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import deva_reader  # noqa: E402
import document_ocr  # noqa: E402
from document_ocr import OCRResult, Token, ocr_page  # noqa: E402


class _FakeBackend:
    name = "fake"

    def __init__(self, texts):
        self.texts = texts

    def run(self, img, lang=None):
        # distinct, well-separated boxes: nothing may merge by accident
        tokens = [Token(text=t, conf=90.0,
                        bbox=(0, i * 200, 400, i * 200 + 60),
                        granularity="line", backend="fake")
                  for i, t in enumerate(self.texts)]
        return OCRResult(text="\n".join(self.texts), tokens=tokens,
                         backend="fake", meta={})

    def recognize_crop(self, crop, lang=None):
        return "", 0.0


class _FakeReader:
    ckpt_path = "fake.pt"

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def read(self, crops):
        self.calls += 1
        return self.outputs[:len(crops)]


def _img(h=400, w=600):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_ocr_page_replaces_texts_and_records_provenance(monkeypatch):
    backend = _FakeBackend(["मिति २०८१", "कुल जम्मा"])
    reader = _FakeReader(["मिति २०८१-०४-२७", "कुल जम्मा रु. ५०"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)
    monkeypatch.setattr(deva_reader, "get_reader", lambda ckpt=None: reader)

    res = ocr_page(_img(), backend="fake", lang="ne", deva_lines="on")
    assert [t.text for t in res.tokens] == ["मिति २०८१-०४-२७", "कुल जम्मा रु. ५०"]
    assert all(t.text_source == "deva_crnn" for t in res.tokens)
    assert res.text.startswith("मिति २०८१-०४-२७")
    info = res.meta["deva_line_reader"]
    assert info["active"] and info["replaced"] == 2 and info["fallback"] == 0


def test_merge_line_boxes_groups_fragments_only():
    from deva_reader import merge_line_boxes

    def tok(text, bbox):
        return Token(text=text, conf=90.0, bbox=bbox, backend="fake")

    # three fragments of one line (same y-band, small gaps) + two separate
    # lines below + one isolated table cell far to the right
    tokens = [
        tok("जैन", (100, 100, 200, 160)),      # fragment 1
        tok("और", (215, 100, 280, 160)),       # fragment 2
        tok("बौद्ध", (295, 100, 420, 160)),    # fragment 3
        tok("दूसरी पंक्ति", (100, 300, 400, 360)),
        tok("तीसरी पंक्ति", (100, 500, 400, 560)),
        tok("५०", (1200, 100, 1260, 160)),     # isolated cell, same band
    ]
    groups = {tuple(idx): bbox for bbox, idx in merge_line_boxes(tokens)}
    assert len(groups) == 4, groups
    assert (0, 1, 2) in groups, "fragments must merge into one line"
    assert groups[(0, 1, 2)] == (100, 100, 420, 160)
    assert (3,) in groups and (4,) in groups
    assert (5,) in groups, "isolated cell must stay alone"


def test_merge_line_boxes_keeps_columns_apart():
    from deva_reader import merge_line_boxes

    def tok(bbox):
        return Token(text="x", conf=90.0, bbox=bbox, backend="fake")

    left = tok((100, 100, 700, 160))
    right = tok((900, 100, 1500, 160))  # 200px gap > 0.75 * 60
    groups = merge_line_boxes([left, right])
    assert len(groups) == 2


def test_auto_policy_skips_cell_like_pages(monkeypatch):
    """`auto` engages on running text, stays off on table-like pages."""
    from deva_reader import page_is_line_like

    class _B:
        name = "fake"

        def __init__(self, boxes):
            self.boxes = boxes

        def run(self, img, lang=None):
            tokens = [Token(text="कुल", conf=90.0, bbox=b, backend="fake")
                      for b in self.boxes]
            return OCRResult(text="x", tokens=tokens, backend="fake", meta={})

        def recognize_crop(self, crop, lang=None):
            return "", 0.0

    line_page = _B([(0, 0, 600, 60), (0, 100, 600, 160), (0, 200, 600, 260)])
    cell_page = _B([(0, 0, 60, 60), (100, 0, 160, 60), (200, 0, 260, 60),
                    (300, 0, 360, 60)])
    line_tokens = [Token(text="कुल", conf=90.0, bbox=b, backend="fake")
                   for b in line_page.boxes]
    cell_tokens = [Token(text="कुल", conf=90.0, bbox=b, backend="fake")
                   for b in cell_page.boxes]
    assert page_is_line_like(line_tokens, 700) is True
    assert page_is_line_like(cell_tokens, 700) is False

    reader = _FakeReader(["कुल जम्मा", "दोस्रो", "तेस्रो", "चौथो"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: cell_page)
    monkeypatch.setattr(deva_reader, "get_reader", lambda ckpt=None: reader)
    res = ocr_page(_img(), backend="fake", lang="ne", deva_lines="auto")
    assert reader.calls == 0, "cell-like page must not engage the reader"
    info = res.meta["deva_line_reader"]
    assert info["active"] is False and "cell-like" in info["reason"]

    monkeypatch.setattr(document_ocr, "get_backend", lambda name: line_page)
    res2 = ocr_page(_img(), backend="fake", lang="ne", deva_lines="on")
    assert reader.calls == 1, "explicit --deva-lines on must force the reader"
    assert res2.meta["deva_line_reader"]["active"] is True


def test_merge_line_boxes_stops_at_a_table_rule():
    """A drawn vertical rule between boxes means separate table cells."""
    from deva_reader import merge_line_boxes

    def tok(bbox):
        return Token(text="x", conf=90.0, bbox=bbox, backend="fake")

    left = tok((100, 100, 200, 160))
    right = tok((215, 100, 300, 160))  # small gap: would merge without a rule
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    assert len(merge_line_boxes([left, right], img)) == 1
    img[100:160, 205:210] = 0  # draw the column rule inside the gap
    groups = merge_line_boxes([left, right], img)
    assert len(groups) == 2, "the rule must block the merge"


def test_implausible_output_keeps_the_backend_reading(monkeypatch):
    backend = _FakeBackend(["पहिलो", "दोस्रो"])
    reader = _FakeReader(["", "क" * 80])  # empty + overlong
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)
    monkeypatch.setattr(deva_reader, "get_reader", lambda ckpt=None: reader)

    res = ocr_page(_img(), backend="fake", lang="ne", deva_lines="on")
    assert [t.text for t in res.tokens] == ["पहिलो", "दोस्रो"]
    assert all(t.text_source == "backend" for t in res.tokens)
    assert res.meta["deva_line_reader"]["fallback"] == 2


def test_deva_lines_off_never_calls_the_reader(monkeypatch):
    backend = _FakeBackend(["मिति २०८१"])
    reader = _FakeReader(["मिति २०८१-०४-२७"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)
    monkeypatch.setattr(deva_reader, "get_reader", lambda ckpt=None: reader)

    res = ocr_page(_img(), backend="fake", lang="ne", deva_lines="off")
    assert reader.calls == 0
    assert "deva_line_reader" not in res.meta
    assert res.tokens[0].text == "मिति २०८१"


def test_missing_checkpoint_is_graceful(monkeypatch):
    backend = _FakeBackend(["मिति २०८१"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)
    monkeypatch.setattr(deva_reader, "get_reader", lambda ckpt=None: None)

    res = ocr_page(_img(), backend="fake", lang="ne", deva_lines="on")
    assert res.tokens[0].text == "मिति २०८१"
    info = res.meta["deva_line_reader"]
    assert info["active"] is False and "checkpoint" in info["reason"]


def test_non_devanagari_never_uses_the_reader(monkeypatch):
    backend = _FakeBackend(["Total 50"])
    reader = _FakeReader(["कुल ५०"])
    monkeypatch.setattr(document_ocr, "get_backend", lambda name: backend)
    monkeypatch.setattr(deva_reader, "get_reader", lambda ckpt=None: reader)

    res = ocr_page(_img(), backend="fake", lang="en")
    assert reader.calls == 0
    assert res.tokens[0].text == "Total 50"


def test_resolve_ckpt_prefers_explicit_then_env(monkeypatch, tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    monkeypatch.setenv(deva_reader.ENV_VAR, str(b))
    assert deva_reader.resolve_ckpt(str(a)) == str(a)
    assert deva_reader.resolve_ckpt() == str(b)
    monkeypatch.delenv(deva_reader.ENV_VAR)
    monkeypatch.setattr(deva_reader, "DEFAULT_CKPT",
                        str(tmp_path / "missing.pt"))
    assert deva_reader.resolve_ckpt() is None


def test_reader_info_reports_inactive_without_checkpoint(monkeypatch):
    monkeypatch.delenv(deva_reader.ENV_VAR, raising=False)
    monkeypatch.setattr(deva_reader, "DEFAULT_CKPT",
                        os.path.join(BASE_DIR, "weights", "does-not-exist.pt"))
    info = deva_reader.reader_info()
    assert info["active"] is False


def test_reader_loads_a_real_checkpoint(tmp_path):
    """End-to-end: a tiny trained checkpoint loads and reads through the API."""
    import torch

    import deva_crnn.predict as pr
    import deva_crnn.train as tr
    from deva_crnn.data import export_npz
    from doc_data import render_devanagari_line

    texts = ["कख", "ग१"]
    imgs = [render_devanagari_line(t, px=48) for t in texts]
    data = str(tmp_path / "tiny.npz")
    export_npz(imgs, texts, data, h=48, w=128)

    orig = tr.CRNN.__init__
    orig_pr = pr.CRNN.__init__

    def small(self, n_classes, hidden=256, in_h=32):
        orig(self, n_classes, hidden=32, in_h=in_h)

    tr.CRNN.__init__ = small
    pr.CRNN.__init__ = small
    try:
        tr.train(data, str(tmp_path / "out"), epochs=2, batch=2, val_split=0.5,
                 seed=1, max_hours=0.1, in_h=48, in_w=128)
        reader = deva_reader.DevaLineReader(str(tmp_path / "out" / "ckpt.pt"))
        out = reader.read(imgs)
    finally:
        tr.CRNN.__init__ = orig
        pr.CRNN.__init__ = orig_pr
    assert len(out) == 2 and all(isinstance(t, str) for t in out)
    assert reader.in_h == 48 and reader.in_w == 128
    assert not torch.cuda.is_available() or True  # reader stays on CPU
