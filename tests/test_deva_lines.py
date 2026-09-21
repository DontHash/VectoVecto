"""
test_deva_lines.py — synthetic Devanagari line crops for recognizer training (W1).

Lines render through Qt shaping (PIL cannot shape conjuncts/matras), so the GT
is exact by construction. The pool deliberately over-samples digits, dates and
amounts - the fine-tune's headline target.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_data import (build_synthetic_line_dataset,  # noqa: E402
                      render_devanagari_line)


def test_render_devanagari_line_shapes_and_ink():
    img = render_devanagari_line("मिति २०८१-०४-२७", px=48, pad=10)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    assert img.shape[0] >= 48
    ink = int((img.min(axis=2) < 128).sum())
    assert ink > 50, "the line must contain rendered glyph pixels"


def test_render_devanagari_line_is_deterministic():
    a = render_devanagari_line("रु. १२,३४५.५०", px=36)
    b = render_devanagari_line("रु. १२,३४५.५०", px=36)
    assert np.array_equal(a, b)


def test_build_synthetic_line_dataset_writes_labels(tmp_path):
    m = build_synthetic_line_dataset(str(tmp_path / "lines"), n=12, seed=3)
    assert m["kind"] == "synthetic_lines"
    assert len(m["entries"]) == 12
    labels = open(os.path.join(str(tmp_path / "lines"), "labels.tsv"),
                  encoding="utf-8").read().strip().splitlines()
    assert len(labels) == 12
    for line in labels:
        name, text = line.split("\t", 1)
        assert os.path.exists(os.path.join(str(tmp_path / "lines"), "pages", name))
        assert text
    assert m["entries"][0]["text"] == labels[0].split("\t", 1)[1]


def test_synthetic_line_pool_is_digit_rich(tmp_path):
    m = build_synthetic_line_dataset(str(tmp_path / "lines"), n=40, seed=5)
    with_digits = sum(1 for e in m["entries"]
                      if any(c.isdigit() or "\u0966" <= c <= "\u096f"
                             for c in e["text"]))
    assert with_digits >= 20, "at least half the lines must carry digits"
