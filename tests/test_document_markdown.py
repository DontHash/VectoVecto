"""
test_document_markdown.py — Markdown transcript output (per page + combined).

The .md carries the reading-order transcript plus the review queue as a table;
it must stay deterministic (no timestamps) and escape table pipes.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from document_export import (transcript_md_text,  # noqa: E402
                             write_combined_transcript_md, write_transcript_md)
from document_ocr import OCRResult, Token  # noqa: E402


def _flagged_result() -> OCRResult:
    toks = [
        Token(text="note", conf=50, bbox=(0, 100, 40, 120), granularity="word",
              flags=["low_conf"]),
        Token(text="TOTAL | 1200.00", conf=90, bbox=(0, 0, 200, 30),
              granularity="line", flags=["digit_conflict"],
              alt_text="TOTAL 1260.00"),
        Token(text="INVOICE", conf=99, bbox=(0, 200, 100, 220),
              granularity="word"),
    ]
    return OCRResult(text="x", tokens=toks, backend="rapidocr")


def test_md_has_provenance_transcript_and_review_table(tmp_path):
    path = write_transcript_md(str(tmp_path / "page.md"), _flagged_result())
    text = open(path, encoding="utf-8").read()

    assert text.startswith("# Transcript\n\n")
    assert "rapidocr · 3 tokens" in text
    assert "1 low-confidence" in text and "1 digit conflicts" in text
    assert "INVOICE" in text
    assert "## Review queue" in text

    # risk-ranked (the digit conflict first), pipes escaped, alt text kept
    assert text.index("TOTAL \\| 1200.00") < text.index("`note`")
    assert "| `TOTAL \\| 1200.00` | digit_conflict | `TOTAL 1260.00` | 90 |" in text
    assert "| `note` | low_conf | — | 50 |" in text


def test_md_body_is_deterministic():
    assert transcript_md_text(_flagged_result()) == \
        transcript_md_text(_flagged_result())


def test_md_without_flags_omits_the_table():
    res = OCRResult(text="clean", tokens=[
        Token(text="HELLO", conf=99, bbox=(0, 0, 60, 20), granularity="word")],
        backend="rapidocr")
    text = transcript_md_text(res)
    assert "## Review queue" not in text
    assert "HELLO" in text and "0 to review" in text


def test_combined_md_numbers_the_pages(tmp_path):
    results = [
        OCRResult(text=f"PAGE {i}", tokens=[
            Token(text=f"PAGE{i}", conf=99, bbox=(0, 0, 50, 20),
                  granularity="word")], backend="fake")
        for i in range(2)
    ]
    path = write_combined_transcript_md(str(tmp_path / "combined.md"),
                                        results, title="multi")
    text = open(path, encoding="utf-8").read()
    assert text.startswith("# multi\n")
    assert "## Page 1" in text and "## Page 2" in text
    assert text.index("PAGE0") < text.index("PAGE1")
