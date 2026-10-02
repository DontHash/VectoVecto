"""
test_document_corrections.py — the verify -> fix -> re-export loop
(docs/CORRECTIONS.md).

Contracts under test: matching is index-first with an original/bbox cross-check
so a stale file can never rewrite the wrong token; unmatched entries are
reported, never silently dropped; changed vs confirmed provenance; corrected
artifacts never overwrite the originals; the training zip contains ONLY marked
items (no crops when nothing is marked).
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document.corrections import (Correction, apply_and_export,  # noqa: E402
                                             apply_corrections,
                                             build_corrections_zip,
                                             corrections_record,
                                             load_corrections,
                                             write_corrections_record)
from veriscript.document.ocr import OCRResult, Token  # noqa: E402


def _result(tokens):
    return OCRResult(text="\n".join(t.text for t in tokens), tokens=tokens,
                     backend="fake", meta={})


def _tok(text, bbox=(10, 10, 110, 50), flags=("low_conf",)):
    return Token(text=text, conf=80.0, bbox=bbox, backend="fake",
                 flags=list(flags))


def test_load_corrections_accepts_list_path_and_kind(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps([
        {"index": 1, "corrected": "X", "bbox": [0, 0, 10, 10]}
    ]), encoding="utf-8")
    doc = load_corrections(str(path))
    assert doc.for_page("anything")[0].corrected == "X"

    with pytest.raises(ValueError):
        load_corrections({"kind": "something-else", "pages": {}})
    with pytest.raises(ValueError):
        load_corrections({"pages": {"p": [{"corrected": "no index"}]}})
    with pytest.raises(ValueError):
        load_corrections({"pages": {"p": [{"index": i, "corrected": "x"}
                                          for i in range(501)]}})


def test_apply_changed_records_provenance():
    good = _tok("२०८१-०४-३९")
    other = _tok("कुल जम्मा")
    res = _result([good, other])
    stats = apply_corrections(res, [
        Correction(index=0, corrected="२०८१-०४-३२",
                   bbox=(10, 10, 110, 50), original="२०८१-०४-३९")])

    assert stats["changed"] == 1 and stats["reviewed"] == 1
    assert good.text == "२०८१-०४-३२"
    assert good.orig_text == "२०८१-०४-३९"
    assert good.text_source == "human" and good.corrected_by == "human"
    assert good.flags == []
    assert res.text == "२०८१-०४-३२\nकुल जम्मा"


def test_apply_confirmed_keeps_text_and_clears_the_queue():
    tok = _tok("मिति २०८१")
    res = _result([tok])
    stats = apply_corrections(res, [
        Correction(index=0, corrected="मिति २०८१", action="confirmed",
                   bbox=(10, 10, 110, 50), original="मिति २०८१")])

    assert stats["confirmed"] == 1 and stats["changed"] == 0
    assert tok.text == "मिति २०८१"
    assert tok.orig_text is None
    assert tok.flags == [] and tok.corrected_by == "human"


def test_stale_index_with_mismatched_original_is_skipped_not_misapplied():
    wrong = _tok("कुल जम्मा")
    res = _result([wrong])
    stats = apply_corrections(res, [
        Correction(index=0, corrected="२०८१-०४-३२",
                   bbox=(5000, 5000, 5100, 5040), original="२०८१-०४-३९")])

    assert stats["skipped"] == 1 and stats["skipped_indices"] == [0]
    assert wrong.text == "कुल जम्मा" and wrong.orig_text is None


def test_bbox_fallback_matches_when_index_is_stale():
    target = _tok("मिति २०८१-०४-२७", bbox=(100, 100, 300, 140))
    res = _result([_tok("other", bbox=(0, 0, 50, 50)), target])
    stats = apply_corrections(res, [
        Correction(index=99, corrected="मिति २०८१-०४-२८",
                   bbox=(100, 95, 300, 145), original="मिति २०८१-०४-२७")])

    assert stats["changed"] == 1
    assert target.text == "मिति २०८१-०४-२८"


def test_apply_and_export_writes_corrected_files_only(tmp_path):
    img = np.full((120, 400, 3), 255, np.uint8)
    tok = _tok("२०८१-०४-३९", bbox=(20, 40, 180, 80))
    res = _result([tok])
    out = str(tmp_path / "out")

    applied = apply_and_export(
        out, "corrected", img, res,
        [Correction(index=0, corrected="२०८१-०४-३२",
                    bbox=(20, 40, 180, 80), original="२०८१-०४-३९")],
        make_overlay=False)

    assert applied["stats"]["changed"] == 1
    assert os.path.isfile(os.path.join(out, "corrected.pdf"))
    assert os.path.isfile(os.path.join(out, "corrected.txt"))
    payload = json.load(open(os.path.join(out, "corrected.json"),
                             encoding="utf-8"))
    entry = payload["tokens"][0]
    assert entry["text"] == "२०८१-०४-३२"
    assert entry["original_text"] == "२०८१-०४-३९"
    assert entry["text_source"] == "human"
    assert entry["corrected_by"] == "human"
    assert payload["review"] == []

    record_path = write_corrections_record(
        os.path.join(out, "corrections.json"),
        corrections_record("page_x", [Correction(index=0, corrected="२०८१-०४-३२")],
                           applied["stats"]))
    record = json.load(open(record_path, encoding="utf-8"))
    assert record["page"] == "page_x" and record["stats"]["changed"] == 1


def test_export_zip_contains_only_marked_items(tmp_path):
    img = np.full((120, 400, 3), 255, np.uint8)
    corrections = [
        Correction(index=0, corrected="१०", bbox=(20, 40, 80, 80),
                   original="२०", action="changed"),
        Correction(index=1, corrected="२०", bbox=(120, 40, 180, 80),
                   original="२०", action="confirmed"),
        Correction(index=2, corrected="३०", bbox=(220, 40, 280, 80),
                   original="३०", action="confirmed"),
    ]
    zip_path = str(tmp_path / "export.zip")
    build_corrections_zip(zip_path, img, "page_a", corrections, share_indices=[0, 1])

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        assert "README.txt" in names and "corrections.json" in names
        assert "labels.tsv" in names
        assert "crops/page_a_0000.png" in names
        assert "crops/page_a_0001.png" in names
        assert "crops/page_a_0002.png" not in names
        labels = z.read("labels.tsv").decode("utf-8")
        assert "page_a_0000.png\t१०" in labels
        assert "page_a_0001.png\t२०" in labels
        assert "page_a_0002" not in labels
        payload = json.loads(z.read("corrections.json").decode("utf-8"))
        assert payload["shared"] == ["page_a_0000.png", "page_a_0001.png"]


def test_export_zip_with_nothing_marked_has_no_crops(tmp_path):
    img = np.full((120, 400, 3), 255, np.uint8)
    zip_path = str(tmp_path / "empty.zip")
    build_corrections_zip(zip_path, img, "page_a", [
        Correction(index=0, corrected="१०", bbox=(20, 40, 80, 80))
    ], share_indices=[])
    with zipfile.ZipFile(zip_path) as z:
        assert not [n for n in z.namelist() if n.startswith("crops/")]
        assert z.read("labels.tsv") == b""
