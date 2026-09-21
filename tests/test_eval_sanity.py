"""
test_eval_sanity.py — unlabeled behavior audit (fake pipeline, no OCR dep).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from eval_sanity import run_audit  # noqa: E402
from harvest_archive_nepali import _pick_pdf  # noqa: E402


class _Tok:
    def __init__(self, flags):
        self.flags = flags


class _Res:
    def __init__(self, meta, n_tokens=3):
        self.meta = meta
        self.ocr = type("O", (), {"tokens": [_Tok([]) for _ in range(n_tokens)]})()


def _pages_dir(tmp_path):
    entries = []
    for i in range(2):
        p = tmp_path / f"page_{i}.png"
        doc_data.imwrite_safe(str(p), np.full((30, 40, 3), 255, dtype=np.uint8))
        entries.append({"id": f"page_{i}", "_image_path": str(p)})
    return entries


def test_run_audit_collects_behavior(tmp_path):
    entries = _pages_dir(tmp_path)

    def fake_pipeline(img, **kw):
        return _Res({"auto_rotate": 0, "orientation_evidence": {"source": "upright"},
                     "orientation_suspect": False, "reading_order_splits": 0,
                     "reading_order_changed": False, "seconds": 1.5}, n_tokens=4)

    res = run_audit(entries, fake_pipeline, lang="ne")
    s = res["summary"]
    assert s["pages"] == 2
    assert s["identity_rate"] == 1.0
    assert s["suspect_rate"] == 0.0
    assert s["tokens_median"] == 4
    assert s["seconds_median"] == 1.5


def test_run_audit_counts_suspects_and_changes(tmp_path):
    entries = _pages_dir(tmp_path)
    state = {"i": 0}

    def fake_pipeline(img, **kw):
        state["i"] += 1
        changed = state["i"] == 1
        return _Res({"auto_rotate": 0, "orientation_suspect": changed,
                     "reading_order_changed": changed, "seconds": 2.0})

    res = run_audit(entries, fake_pipeline)
    assert res["summary"]["suspect_rate"] == 0.5
    assert res["summary"]["identity_rate"] == 0.5


def test_pick_pdf_prefers_text_pdf():
    files = [{"name": "thumb.jpg", "format": "JPEG Thumb"},
             {"name": "scan.pdf", "format": "Image Container PDF"},
             {"name": "text.pdf", "format": "Text PDF"}]
    assert _pick_pdf(files) == "text.pdf"
    assert _pick_pdf([{"name": "a.pdf", "format": "Something"}]) == "a.pdf"
    assert _pick_pdf([{"name": "a.jpg", "format": "JPEG"}]) is None
