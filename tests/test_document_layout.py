"""
test_document_layout.py — conservative reading-order tests (pure, no OCR).

Contract: repair *confident* column layouts; otherwise preserve engine order
(both supported backends already emit rows top-down, and aggressive row
re-grouping measurably hurt single-column receipts).
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from document_layout import row_major, sort_reading_order  # noqa: E402
from document_ocr import Token  # noqa: E402


def T(text, x0, y0, x1, y1, conf=99.0, gran="line"):
    return Token(text=text, conf=conf, bbox=(x0, y0, x1, y1), granularity=gran)


def column_pairs(n, start_y=40, step=80, left_x0=40, right_x0=400):
    """Row-major engine order (L1, R1, L2, R2, ...) with ragged right edges."""
    toks = []
    for i in range(n):
        y = start_y + i * step
        lx1 = 300 - (i * 53) % 90
        rx1 = right_x0 + 300 - (i * 37) % 110
        toks.append(T(f"L{i + 1} line", left_x0, y, lx1, y + 30))
        toks.append(T(f"R{i + 1} line", right_x0, y, rx1, y + 30))
    return toks


def test_two_columns_read_column_major():
    toks = column_pairs(7)
    out = sort_reading_order(toks)
    assert [t.text for t in out] == (
        [f"L{i} line" for i in range(1, 8)] +
        [f"R{i} line" for i in range(1, 8)])


def test_small_label_value_table_stays_engine_order():
    # 3 clean-gutter rows: label/value table, must NOT be columnized.
    toks = column_pairs(3)
    out = sort_reading_order(toks)
    assert [t.text for t in out] == [t.text for t in toks]


def test_min_column_tokens_override():
    toks = column_pairs(2)
    texts = [t.text for t in sort_reading_order(toks, min_column_tokens=1)]
    assert texts == ["L1 line", "L2 line", "R1 line", "R2 line"]


def test_right_aligned_value_column_stays_engine_order():
    # Receipt totals: labels left, amounts right-aligned (x1 constant) ->
    # row-major order must be kept even though the gutter is clean and long.
    toks = []
    for i in range(7):
        y = 40 + i * 80
        toks.append(T(f"label{i}", 40, y, 250 - (i * 31) % 70, y + 30))
        toks.append(T(f"{i}.00", 660 + (i % 3) * 10, y, 700, y + 30))
    out = sort_reading_order(toks)
    assert [t.text for t in out] == [t.text for t in toks]
    assert out[0].text == "label0" and out[1].text == "0.00"


def test_full_width_title_then_columns():
    toks = column_pairs(6, start_y=200)
    toks.append(T("TITLE SPANS WIDTH", 40, 40, 700, 80))
    out = sort_reading_order(toks)
    texts = [t.text for t in out]
    assert texts[0] == "TITLE SPANS WIDTH"
    assert texts[1:7] == [f"L{i} line" for i in range(1, 7)]
    assert texts[7:] == [f"R{i} line" for i in range(1, 7)]


def test_wide_title_separates_band_even_when_close_to_columns():
    # Title bottom y=120, first row y=140: gap far below the band threshold,
    # so only the full-width rule can keep the columns band clean.
    toks = column_pairs(7, start_y=140)
    toks.append(T("TITLE SPANS WIDTH", 40, 40, 700, 120))
    out = sort_reading_order(toks)
    texts = [t.text for t in out]
    assert texts[0] == "TITLE SPANS WIDTH"
    assert texts[1:8] == [f"L{i} line" for i in range(1, 8)]
    assert texts[8:] == [f"R{i} line" for i in range(1, 8)]


def test_wide_token_sharing_a_row_is_not_a_separator():
    from document_layout import _content_width, _standalone_wide_ids

    wide = T("WIDE LINE", 0, 0, 1000, 30)
    same_row = T("value", 700, 5, 900, 25)
    solo = T("title", 0, -100, 1000, -70)
    toks = [wide, same_row]
    assert _standalone_wide_ids(toks, _content_width(toks)) == set()
    toks = [wide, same_row, solo]
    ids = _standalone_wide_ids(toks, _content_width(toks))
    assert id(solo) in ids and id(wide) not in ids


def test_non_column_preserves_engine_order():
    ordered = [
        T("first", 40, 40, 300, 70),
        T("second", 40, 120, 300, 150),
        T("third", 40, 200, 300, 230),
    ]
    assert [t.text for t in sort_reading_order(ordered)] == \
        ["first", "second", "third"]
    scrambled = [ordered[2], ordered[0], ordered[1]]
    assert [t.text for t in sort_reading_order(scrambled)] == \
        ["third", "first", "second"], "engine order is preserved when unsure"


def test_words_in_one_line_stay_left_to_right():
    toks = [
        T("world", 140, 40, 220, 65, gran="word"),
        T("hello", 40, 40, 120, 65, gran="word"),
    ]
    out = row_major(toks)
    assert [t.text for t in out] == ["hello", "world"]


def test_empty_input():
    assert sort_reading_order([]) == []


def test_thin_label_strip_stays_engine_order():
    # sroie_00003 regression: a 6-token label strip (24% of content width)
    # beside a value column passes coverage/raggedness but is not a column.
    toks = []
    for i in range(6):
        y = 40 + i * 40
        toks.append(T(f"label{i}", 20, y, 120 - (i * 13) % 30, y + 22))
    for i in range(15):
        y = 30 + i * 18
        toks.append(T(f"value item {i}", 145, y, 440 - (i * 29) % 120, y + 16))
    out = sort_reading_order(toks)
    assert [t.text for t in out] == [t.text for t in toks], \
        "thin label strips must never be columnized"


def test_gutter_tolerates_box_padding():
    # Real-page condition: OCR boxes bleed a few px into the gutter. The
    # coverage rule (<=5% crossings) must still columnize.
    toks = column_pairs(7)
    toks[0] = T("L1 line", 40, 40, 330, 70, gran="line")  # pads into the gutter
    out = sort_reading_order(toks)
    assert [t.text for t in out] == (
        [f"L{i} line" for i in range(1, 8)] +
        [f"R{i} line" for i in range(1, 8)])


def test_pipeline_applies_reading_order(monkeypatch):
    import numpy as np
    import document_pipeline as dp
    from document_ocr import OCRResult

    def fake_restore(img, **kw):
        return {"display_bgr": img, "skew_angle": 0.0,
                "debug": {"skew_angle": 0.0}}

    toks = column_pairs(7)
    monkeypatch.setattr(dp, "restore_document", fake_restore)
    monkeypatch.setattr(dp, "pick_backend", lambda b=None: "rapidocr")
    monkeypatch.setattr(dp, "ocr_page",
                        lambda img, **kw: OCRResult(text="", tokens=list(toks),
                                                    backend="rapidocr", meta={}))
    monkeypatch.setattr(dp, "compare_digit_streams", lambda a, b: 0)

    page = np.full((700, 800, 3), 255, dtype=np.uint8)
    res = dp.run_document_pipeline(page, backend="rapidocr")
    expected = [f"L{i} line" for i in range(1, 8)] + \
               [f"R{i} line" for i in range(1, 8)]
    assert [t.text for t in res.ocr.tokens] == expected
    assert res.ocr.text.splitlines() == expected
    assert res.meta["reading_order"] is True

    res2 = dp.run_document_pipeline(page, backend="rapidocr", reading_order=False)
    assert [t.text for t in res2.ocr.tokens] == [t.text for t in toks]
    assert res2.meta["reading_order"] is False
