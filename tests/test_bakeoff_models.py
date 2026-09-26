"""
test_bakeoff_models.py — bake-off registry, line extraction, scoring, metrics.

No heavy deps: candidate availability checks are exercised, but no model is
downloaded here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import bakeoff_models  # noqa: E402
import doc_data  # noqa: E402
from doc_metrics import bag_stats, digit_exact, digit_string  # noqa: E402
from eval_models import iter_heidata_lines, score_lines  # noqa: E402


def test_registry_and_list_candidates():
    names = set(bakeoff_models.REGISTRY)
    assert {"rapidocr", "tesseract-nep", "trocr", "glmocr", "bodhan",
            "qwen3vl", "qwen3vl-8b-4bit", "qwen3vl-8b-4bit-rt",
            "qwen3vl-4b", "qwen3vl-4b-4bit"} <= names
    listed = bakeoff_models.list_candidates()
    assert len(listed) == len(bakeoff_models.REGISTRY)
    for name, kind, lic, ok, why in listed:
        assert isinstance(ok, bool) and isinstance(why, str)
        assert kind in ("lines", "pages", "both")
        assert lic and lic != "?"


def test_load_lines_source_deva_real_lines(tmp_path):
    from eval_models import load_lines_source

    root = tmp_path / "deva_real_lines"
    (root / "lines").mkdir(parents=True)
    (root / "lines" / "a.png").write_bytes(b"x")
    (root / "labels.tsv").write_text("a.png\tकुल १२०.५०\n", encoding="utf-8")
    items = load_lines_source("deva_real_lines", str(root), limit=10)
    assert len(items) == 1
    assert items[0]["id"] == "a.png" and items[0]["gt"] == "कुल १२०.५०"
    assert items[0]["img_path"] == os.path.join(str(root), "lines", "a.png")


def test_qwen_fit_pixels_caps_area():
    """A born-digital page (~8.4 Mpx) must be downscaled to the model budget.

    Regression: native-res pages OOMed the V100 vision tower (8.16 GB alloc).
    """
    cand = bakeoff_models.REGISTRY["qwen3vl"]
    page = np.zeros((3301, 2550, 3), dtype=np.uint8)
    out = cand._fit_pixels(page)
    assert out.shape[0] * out.shape[1] <= cand.max_pixels
    assert out.shape[2] == 3
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    assert cand._fit_pixels(small) is small


def test_digit_string_and_exact():
    assert digit_string("रु. १२०.५०") == "120.50"
    assert digit_string("total: 1,200.00") == "1,200.00"
    assert digit_string("no digits here") == ""
    assert digit_exact("कुल १२०.५०", "total 120.50")
    assert not digit_exact("१२०.५०", "120.00")


def test_bag_stats_hallucination_proxy():
    gt = "कुल योग 120.50"
    s = bag_stats(gt, "कुल योग 120.50")
    assert s["miss_rate"] == 0.0 and s["invented_rate"] == 0.0
    s2 = bag_stats(gt, "कुल 999.99")
    assert s2["invented_tokens"] == 1
    assert 0 < s2["miss_rate"] <= 1.0
    s3 = bag_stats(gt, "")
    assert s3["miss_rate"] == 1.0 and s3["invented_rate"] == 0.0


def _fake_heidata(tmp_path):
    pages = tmp_path / "pages"
    gt = tmp_path / "gt"
    pages.mkdir()
    gt.mkdir()
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    p = pages / "book_p00.png"
    doc_data.imwrite_safe(str(p), img)
    boxes = [
        {"text": "प्रथम पंक्ति", "bbox": [10, 10, 300, 40]},
        {"text": "कुल १२०.५०", "bbox": [10, 50, 300, 80]},
        {"text": "शुद्ध पंक्ति", "bbox": [10, 90, 300, 120]},
    ]
    doc_data.imwrite_safe(str(pages / "book_p00.png"), img)
    (gt / "book_p00.txt").write_text("\n".join(b["text"] for b in boxes),
                                     encoding="utf-8")
    (gt / "book_p00.boxes.json").write_text(json.dumps(boxes, ensure_ascii=False),
                                            encoding="utf-8")
    manifest = {"kind": "real_pages", "entries": [{
        "id": "book_p00", "clean": "pages/book_p00.png",
        "degraded": "pages/book_p00.png", "gt": "gt/book_p00.txt",
        "boxes": "gt/book_p00.boxes.json", "real": True}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(tmp_path)


def test_iter_heidata_lines_deterministic_digit_only(tmp_path):
    d = _fake_heidata(tmp_path)
    items = iter_heidata_lines(d, limit=10)
    assert [it["gt"] for it in items] == ["कुल १२०.५०"]
    again = iter_heidata_lines(d, limit=10)
    assert [it["id"] for it in again] == [it["id"] for it in items]


def test_score_lines_with_fake_candidate(tmp_path):
    d = _fake_heidata(tmp_path)
    items = iter_heidata_lines(d, limit=10)

    class Fake(bakeoff_models.Candidate):
        name = "fake"

        def recognize_lines(self, ctx, crops, lang=None):
            return ["कुल १२०.५०"] * len(crops)

    res = score_lines(Fake(), None, items, lang=None)
    s = res["summary"]
    assert s["lines"] == 1
    assert s["cer"] == 0.0
    assert s["cer_median"] == 0.0
    assert s["catastrophic_rate"] == 0.0
    assert s["digit_exact_rate"] == 1.0
    assert s["exact_match"] == 1.0

    class Junk(bakeoff_models.Candidate):
        name = "junk"

        def recognize_lines(self, ctx, crops, lang=None):
            return [""] * len(crops)

    res2 = score_lines(Junk(), None, items, lang=None)
    assert res2["summary"]["digit_exact_rate"] == 0.0
    assert res2["summary"]["empty_outputs"] == 1
