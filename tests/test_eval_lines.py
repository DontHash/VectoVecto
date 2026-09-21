"""
test_eval_lines.py — recognition-only line eval (fake recognizer, no OCR dep).
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from eval_lines import make_contact_sheet, score_lines, write_review_csv  # noqa: E402


def _fake_lines_dir(tmp_path):
    pages = tmp_path / "pages"
    gt = tmp_path / "gt"
    pages.mkdir()
    gt.mkdir()
    texts = ["first line of nepali text", "दोस्रो पंक्ति", "Rs 1234.00"]
    entries = []
    for i, text in enumerate(texts):
        img = np.full((60, 400, 3), 255, dtype=np.uint8)
        cv2.putText(img, f"row{i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2)
        img_path = pages / f"line_{i:05d}.png"
        gt_path = gt / f"line_{i:05d}.txt"
        cv2.imwrite(str(img_path), img)
        gt_path.write_text(text, encoding="utf-8")
        entries.append({"id": f"line_{i:05d}", "image": f"pages/line_{i:05d}.png",
                        "gt": f"gt/line_{i:05d}.txt", "real": True})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"kind": "lines", "source": "fake", "split": "test",
                    "entries": entries}), encoding="utf-8")
    return entries


def test_score_lines_perfect_and_wrong(tmp_path):
    _fake_lines_dir(tmp_path)
    manifest = doc_data.load_line_dataset(str(tmp_path))
    entries = manifest["entries"]
    gts = [open(e["_gt_path"], encoding="utf-8").read() for e in entries]

    state = {"i": 0}

    def perfect(img):
        i = state["i"]
        state["i"] += 1
        return gts[i], 99.0

    res = score_lines(entries, perfect)
    assert res["summary"]["lines"] == 3
    assert res["summary"]["cer"] == 0.0
    assert res["summary"]["exact_match"] == 1.0
    assert res["summary"]["digit_cer"] == 0.0

    def empty(img):
        return "", 0.0

    res2 = score_lines(entries, empty)
    assert res2["summary"]["cer"] == 1.0
    assert res2["summary"]["empty_outputs"] == 3
    assert res2["summary"]["exact_match"] == 0.0


def test_score_lines_digit_metric_only_on_digit_rows(tmp_path):
    _fake_lines_dir(tmp_path)
    entries = doc_data.load_line_dataset(str(tmp_path))["entries"]

    def recognize(img):
        return "wrong output", 50.0

    res = score_lines(entries, recognize)
    assert res["summary"]["digit_lines"] == 1
    assert res["summary"]["digit_cer"] == pytest.approx(1.0)


def test_contact_sheet_and_review_csv(tmp_path):
    _fake_lines_dir(tmp_path)
    entries = doc_data.load_line_dataset(str(tmp_path))["entries"]

    def recognize(img):
        return "partial", 50.0

    res = score_lines(entries, recognize)
    for r in res["rows"]:
        r["_image_path"] = next(e["_image_path"] for e in entries if e["id"] == r["id"])
    sheet = make_contact_sheet(res["rows"], str(tmp_path / "sheet.png"), n=3)
    assert os.path.exists(sheet)
    csv_path = write_review_csv(res["rows"], str(tmp_path / "review.csv"))
    lines = open(csv_path, encoding="utf-8").read().splitlines()
    assert lines[0] == "id,cer,gt,ocr"
    assert len(lines) == 4
