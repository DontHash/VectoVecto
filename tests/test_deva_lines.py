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


def test_multi_font_pool_renders_distinct_lines():
    from doc_data import available_deva_font_specs, synthesize_deva_lines
    specs = available_deva_font_specs()
    assert specs, "at least one Devanagari font must be installed"
    imgs, texts = synthesize_deva_lines(24, seed=7, fonts=specs, jitter=True)
    assert len(imgs) == 24 and all(t for t in texts)
    paths = {s["path"] for s in specs}
    if len(paths) >= 2:
        a = render_devanagari_line("नेपाल सरकार", px=40,
                                   spec=specs[0])
        b = render_devanagari_line("नेपाल सरकार", px=40,
                                   spec=[s for s in specs
                                         if s["path"] != specs[0]["path"]][0])
        assert a.shape != b.shape or not np.array_equal(a, b), \
            "different font files must render differently"


def test_letter_spacing_and_stretch_change_the_line():
    a = render_devanagari_line("मिति २०८१-०४-२७", px=40)
    b = render_devanagari_line("मिति २०८१-०४-२७", px=40, letter_spacing=1.5,
                               stretch=110)
    assert a.shape != b.shape or not np.array_equal(a, b)


def test_cell_pool_is_short_and_digit_rich():
    from doc_data import synthesize_deva_lines
    imgs, texts = synthesize_deva_lines(60, seed=11, cell_frac=1.0)
    assert len(imgs) == 60
    assert all(len(t) <= 8 for t in texts), "cells must stay short"
    with_digits = sum(1 for t in texts if any(c.isdigit() for c in t))
    assert with_digits >= 45, "cell pool must be digit-heavy"
    # short cells must still render as ink-bearing crops
    assert all(int((im.min(axis=2) < 128).sum()) > 10 for im in imgs)


def test_punctuation_parity_with_the_real_corpus():
    """Parenthesized numbers and letterpress formats must be in the pool.

    Measured gap (W1 attempt 3b): `(`/`)` occur 1 line in 23 in the real
    letterpress GT but 1 in 300 in training; the recognizer read `)` as `१`.
    """
    from doc_data import synthesize_deva_lines
    _imgs, texts = synthesize_deva_lines(600, seed=13)
    joined = "\n".join(texts)
    assert "(" in joined and ")" in joined, "parenthesized numbers missing"
    assert "[" in joined and "]" in joined, "bracketed numbers missing"
    assert "सन्" in joined, "letterpress date format missing"
    assert "।।" in joined or "॥" in joined, "danda variants missing"
    assert "नं०" in joined or "नं॰" in joined
    # rare characters present in the real GT must appear at all
    for ch in ("़", "ॉ", "॰", '"', "="):
        assert ch in joined, f"rare GT char {ch!r} missing from the pool"


def test_pool_contains_long_lines_matching_real_widths():
    """Real letterpress lines are wide; the pool must cover 40-60 chars."""
    from doc_data import synthesize_deva_lines
    _imgs, texts = synthesize_deva_lines(400, seed=17)
    long_lines = [t for t in texts if len(t) >= 40]
    assert len(long_lines) >= 60, "at least ~15% of lines must be long"
    assert all(len(t) <= 60 for t in long_lines)
    assert any(any(c.isdigit() for c in t) for t in long_lines), \
        "long lines must carry digits too"
