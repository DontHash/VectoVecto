"""
test_eval_anchor.py — worksheet selection + artifacts (N5 anchor).

The anchor has three machine readings per page (text layer / RapidOCR /
bodhan). Pages above the agreement bar are consensus-anchored; the rest go to
a budgeted worksheet, stratified across documents, pre-filled with the
RapidOCR reading so the human only edits errors.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from eval_anchor import (mark_disagreements, render_html,  # noqa: E402
                         select_worksheet, write_verified_stubs)


def _rows():
    return [
        {"page": "a_p0", "doc": "a", "agreement_rate": 0.99,
         "readings": {"rapidocr": "एक"}},
        {"page": "a_p1", "doc": "a", "agreement_rate": 0.80,
         "readings": {"rapidocr": "दुई"}},
        {"page": "a_p2", "doc": "a", "agreement_rate": 0.70,
         "readings": {"rapidocr": "तीन"}},
        {"page": "b_p0", "doc": "b", "agreement_rate": 0.85,
         "readings": {"rapidocr": "चार"}},
        {"page": "b_p1", "doc": "b", "agreement_rate": 0.60,
         "readings": {"rapidocr": "पाँच"}},
        {"page": "c_p0", "doc": "c", "agreement_rate": 0.90,
         "readings": {"rapidocr": "छ"}},
    ]


def test_select_worksheet_filters_consensus_and_respects_budget():
    consensus, picked = select_worksheet(_rows(), threshold=0.95, budget=3)
    assert consensus == ["a_p0"]
    assert len(picked) == 3
    assert "a_p0" not in picked


def test_select_worksheet_is_stratified_and_deterministic():
    _, picked = select_worksheet(_rows(), threshold=0.95, budget=4)
    # round-robin over docs: lowest agreement first within each doc
    assert picked == ["a_p2", "b_p1", "c_p0", "a_p1"]


def test_write_verified_stubs_prefills_and_never_clobbers(tmp_path):
    _, picked = select_worksheet(_rows(), threshold=0.95, budget=2)
    written = write_verified_stubs(_rows(), str(tmp_path), picked)
    assert len(written) == 2
    assert (tmp_path / "a_p2.txt").read_text(encoding="utf-8").strip() == "तीन"
    (tmp_path / "a_p2.txt").write_text("मानिसले सच्याएको", encoding="utf-8")
    write_verified_stubs(_rows(), str(tmp_path), picked)
    assert (tmp_path / "a_p2.txt").read_text(encoding="utf-8") == "मानिसले सच्याएको"


def test_mark_disagreements_marks_only_foreign_tokens():
    html = mark_disagreements("नेपालको संविधान", "नेपालको संविधानन्")
    assert "<mark>संविधान</mark>" in html
    assert "<mark>नेपालको</mark>" not in html


def test_mark_disagreements_ignores_punctuation():
    html = mark_disagreements("धारा १,", "धारा १")
    assert "<mark>" not in html


def test_score_ignores_stale_prefills_outside_worksheet(tmp_path):
    import json

    from eval_anchor import score

    readings = {
        "threshold": 0.95, "budget": 1, "consensus": [],
        "worksheet": ["a_p1"],
        "rows": [
            {"page": "a_p1", "doc": "a", "agreement_rate": 0.8,
             "readings": {"text_layer": "नमस्ते", "rapidocr": "नमस्ते",
                          "bodhan": "नमसते"}},
            {"page": "a_p2", "doc": "a", "agreement_rate": 0.8,
             "readings": {"text_layer": "नमस्ते", "rapidocr": "नमस्ते",
                          "bodhan": "नमसते"}},
        ],
    }
    path = tmp_path / "readings.json"
    path.write_text(json.dumps(readings, ensure_ascii=False), encoding="utf-8")
    verified = tmp_path / "verified"
    verified.mkdir()
    (verified / "a_p1.txt").write_text("नमस्ते", encoding="utf-8")
    (verified / "a_p2.txt").write_text("नमस्ते", encoding="utf-8")
    result = score(str(verified), str(path))
    assert [p["page"] for p in result["per_page"]] == ["a_p1"]
    assert result["skipped_stale"] == ["a_p2"]


def test_render_html_embeds_page_image_and_readings(tmp_path):
    import cv2

    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:, :, 1] = 180
    img_path = str(tmp_path / "page.png")
    cv2.imwrite(img_path, img)
    rows = [{"page": "a_p1", "doc": "a", "agreement_rate": 0.8,
             "image": img_path,
             "readings": {"text_layer": "नमस्ते", "rapidocr": "नमस्ते",
                          "bodhan": "नमसते"},
             "agreement": {"disagreements": ["नमसते"]}}]
    out = str(tmp_path / "worksheet.html")
    render_html(rows, out)
    html = open(out, encoding="utf-8").read()
    assert "a_p1" in html
    assert "नमसते" in html
    assert "data:image/png;base64," in html
