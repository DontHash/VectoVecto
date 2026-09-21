"""
test_real_lines.py — Gemini-labeled real-line mining (W1 attempt 2).

The DP alignment is the safety-critical piece: a crossed match would attach
the wrong label to a crop and poison training. These tests pin monotonicity,
the similarity bar and the label filters with fake OCR/Gemini.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import label_real_lines as lrl  # noqa: E402


class _Tok:
    def __init__(self, text, bbox):
        self.text = text
        self.bbox = bbox


class _Rec:
    def __init__(self, tokens):
        self.tokens = tokens


def _img(h=200, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_align_lines_is_monotone_and_bars_low_similarity():
    ocr = ["नेपाल सरकार", "मिति २०८१-०४-२७", "कुल जम्मा"]
    gem = ["नेपाल सरकार", "मिति २०८१-०४-२७", "कुल जम्मा"]
    pairs = lrl.align_lines(ocr, gem)
    assert [p[0] for p in pairs] == [0, 1, 2]
    assert [p[1] for p in pairs] == [0, 1, 2]
    assert all(p[2] == 1.0 for p in pairs)

    # a Gemini line that shares nothing must not be matched
    pairs2 = lrl.align_lines(["नेपाल सरकार"], ["पूर्ण रूपमा फरक"])
    assert pairs2 == []


def test_align_lines_skips_a_missing_line_without_crossing():
    ocr = ["पहिलो हरफ", "दोस्रो हरफ", "तेस्रो हरफ"]
    gem = ["पहिलो हरफ", "तेस्रो हरफ"]  # Gemini dropped the middle line
    pairs = lrl.align_lines(ocr, gem)
    assert [(i, j) for i, j, _s in pairs] == [(0, 0), (2, 1)]


def test_label_page_keeps_only_clean_pairs():
    img = _img()
    tokens = [_Tok("मिति २०८१-०४-२७", (10, 10, 200, 40)),
              _Tok("कुल जम्मा रु. १,२३४.५०", (10, 50, 260, 80)),
              _Tok("शीर्षक", (10, 90, 120, 120))]

    def fake_ocr(_img):
        return _Rec(tokens)

    def fake_transcribe(_client, _img, model=None, strips=None):
        return ("मिति २०८१-०४-२८\n"          # 1 char off -> still matches
                "कुल जम्मा रु. १,२३४.५०\n"
                "पूर्ण फरक हरफ\n")            # no match -> dropped

    row = lrl.label_page(None, img, "p0", "fake", 1, 0.6,
                         ocr_fn=fake_ocr, transcribe_fn=fake_transcribe)
    assert row["matched"] == 2
    texts = [k["text"] for k in row["kept"]]
    assert texts == ["मिति २०८१-०४-२८", "कुल जम्मा रु. १,२३४.५०"]
    assert all(k["crop"].shape[2] == 3 and k["crop"].size > 0
               for k in row["kept"])


def test_label_page_drops_non_devanagari_and_overlong():
    img = _img()
    tokens = [_Tok("12345", (0, 0, 100, 30)),
              _Tok("क" * 60, (0, 40, 300, 70))]

    def fake_ocr(_img):
        return _Rec(tokens)

    def fake_transcribe(_client, _img, model=None, strips=None):
        return "12345\n" + ("क" * 60) + "\n"

    row = lrl.label_page(None, img, "p0", "fake", 1, 0.6,
                         ocr_fn=fake_ocr, transcribe_fn=fake_transcribe)
    assert row["matched"] == 2 and row["kept"] == [] and row["dropped"] == 2


def test_build_montage_stacks_strips_vertically():
    crops = [np.full((20, 100, 3), 255, np.uint8),
             np.full((30, 80, 3), 200, np.uint8)]
    m = lrl.build_montage(crops, gap=10, pad=8)
    assert m.shape[0] == 20 + 30 + 10 + 16
    assert m.shape[1] == 100 + 16
    assert (m[8:28, 8:108] == 255).all()          # first strip at the top
    assert (m[38:68, 8:88] == 200).all()          # second strip below the gap


def test_transcribe_montage_is_one_to_one():
    crops = [np.full((20, 120, 3), 255, np.uint8) for _ in range(5)]
    hints = ["नेपाल सरकार", "मिति २०८१", "कुल जम्मा", "रु. ५०", "धारा १२"]
    calls = []

    def fake_call(_client, prompt, _img, model, timeout_s=0, retries=0):
        calls.append(prompt)
        return "\n".join(hints)

    orig = lrl._call_gemini
    lrl._call_gemini = fake_call
    try:
        out = lrl.transcribe_montage(object(), crops, hints=hints, rows=3)
    finally:
        lrl._call_gemini = orig
    assert out == hints
    assert len(calls) == 2, "5 crops with rows=3 must take 2 calls"


def test_transcribe_montage_count_mismatch_drops_rather_than_mislabels():
    crops = [np.full((20, 120, 3), 255, np.uint8) for _ in range(3)]
    hints = ["नेपाल सरकार", "मिति २०८१-०४-२७", "कुल जम्मा"]

    def fake_call(_client, prompt, _img, model, timeout_s=0, retries=0):
        return "मिति २०८१-०४-२७\nकुल जम्मा"  # merged: 2 lines for 3 strips

    orig = lrl._call_gemini
    lrl._call_gemini = fake_call
    try:
        out = lrl.transcribe_montage(object(), crops, hints=hints, rows=3)
    finally:
        lrl._call_gemini = orig
    assert out[0] == ""
    assert out[1:] == ["मिति २०८१-०४-२७", "कुल जम्मा"]


def test_label_page_montage_keeps_crops_and_filters():
    img = _img()
    tokens = [_Tok("नेपाल सरकार", (0, 0, 200, 30)),
              _Tok("12345", (0, 40, 100, 60)),
              _Tok("abc def", (0, 70, 100, 90)),
              _Tok("मिति २०८१-०४-२७", (0, 80, 200, 110))]

    def fake_ocr(_img):
        return _Rec(tokens)

    def fake_montage(_client, crops, hints=None, model=None, rows=0):
        return ["नेपाल सरकार", "12345", "abc def", "मिति २०८१-०४-२७"]

    row = lrl.label_page_montage(None, img, "p0", "fake", ocr_fn=fake_ocr,
                                 montage_fn=fake_montage)
    assert len(row["kept"]) == 3 and row["dropped"] == 1
    assert [k["text"] for k in row["kept"]] == ["नेपाल सरकार", "12345",
                                               "मिति २०८१-०४-२७"]
    assert row["dropped_reasons"]["chars"] == 1
    assert all(k["crop"].size > 0 for k in row["kept"])


def test_frozen_pages_reads_the_v2_manifest(tmp_path):
    m = {"entries": [{"source": r"D:\x\a.pdf", "page": 0},
                     {"source": r"D:\x\a.pdf", "page": 1},
                     {"source": r"D:\x\b.pdf", "page": 3}]}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    used = lrl.frozen_pages(str(p))
    assert used == {"a.pdf": {0, 1}, "b.pdf": {3}}


def test_run_resumes_and_writes_labels(tmp_path):
    img = _img()
    tokens = [_Tok("नेपाल सरकार", (0, 0, 200, 30))]

    def fake_ocr(_img):
        return _Rec(tokens)

    def fake_transcribe(_client, _img, model=None, strips=None):
        return "नेपाल सरकार\n"

    out = tmp_path / "real"
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "manifest.json").write_text(json.dumps({"entries": []}),
                                      encoding="utf-8")

    report = {"kind": "real_lines", "model": "fake", "strips": 1,
              "min_sim": 0.6, "dpi": 300,
              "pages": {"a_p000": {"source": "a.pdf", "page": 0,
                                   "ocr_lines": 1, "gemini_lines": 1,
                                   "matched": 1, "kept": 1, "dropped": 0,
                                   "lines": [["a_p000_l000.png", "नेपाल सरकार",
                                              1.0]]}}}
    os.makedirs(str(out), exist_ok=True)
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")

    # no pages to do (frozen manifest empty + no sources) -> just finalizes
    res = lrl.run(str(out), str(v2), per_doc=1, client=object(),
                  ocr_fn=fake_ocr, transcribe_fn=fake_transcribe)
    assert res["summary"]["pages"] == 1
    assert res["summary"]["lines_kept"] == 1
