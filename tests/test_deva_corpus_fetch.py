"""
test_deva_corpus_fetch.py — W-C corpus builder: filters, sampling, manifest.

Network is never touched: `fetch` runs against a local .xz fixture.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import fetch_deva_corpus as fc  # noqa: E402
from export_training_data import load_corpus  # noqa: E402


def _fixture(tmp_path, lines):
    path = tmp_path / "ne.txt.xz"
    path.write_bytes(lzma.compress(("\n".join(lines) + "\n").encode("utf-8")))
    return str(path)


def test_line_ok_filters():
    assert fc.line_ok("नेपाल सरकारले आज बजेट सार्वजनिक गरेको छ")
    assert not fc.line_ok("नेपाल"), "too short"
    assert not fc.line_ok("क " * 40), "too long"
    assert not fc.line_ok("https://example.com नेपाल सरकारको वेबसाइट हो")
    assert not fc.line_ok("hello world this is an english sentence only")
    assert not fc.line_ok("नेपालमा \ufffd समस्या छ")
    assert not fc.line_ok("िक देश"), "invalid combining sequence"
    assert not fc.line_ok("नेपालमा ABC कम्पनीले काम गरेको छ"), "latin letters"
    assert not fc.line_ok("नेपालमा @ कम्पनीले काम गरेको छ"), "charset"


def test_build_filters_and_samples_deterministically(tmp_path):
    lines = [
        "नेपाल सरकारले आज बजेट सार्वजनिक गरेको छ",
        "hello world this is english",
        "यो वर्ष धान उत्पादन बढेको छ",
        "नेपाल",  # too short
        "काठमाडौंमा आज वर्षा भएको छ",
    ]
    src = _fixture(tmp_path, lines)
    a = fc.build(src, limit=2, seed=7)
    b = fc.build(src, limit=2, seed=7)
    assert a["lines"] == b["lines"], "reservoir sampling must be deterministic"
    assert len(a["lines"]) == 2
    assert a["stats"]["source_lines"] == 5
    assert a["stats"]["kept"] == 3
    assert all(fc.line_ok(t) for t in a["lines"])


def test_fetch_with_local_source_writes_manifest(tmp_path):
    src = _fixture(tmp_path, [
        "नेपाल सरकारले आज बजेट सार्वजनिक गरेको छ",
        "यो वर्ष धान उत्पादन बढेको छ",
    ])
    out_dir = str(tmp_path / "out")
    manifest = fc.fetch(out_dir, local=src, limit=10, seed=1)
    out_path = os.path.join(out_dir, manifest["output"]["file"])
    assert manifest["output"]["lines"] == 2
    assert manifest["output"]["sha256"] == hashlib.sha256(
        open(out_path, "rb").read()).hexdigest()
    assert manifest["source"]["sha256"] == hashlib.sha256(
        open(src, "rb").read()).hexdigest()
    assert "CC-100" in manifest["source"]["license"]
    assert os.path.exists(os.path.join(out_dir, fc.MANIFEST_NAME))
    assert len(load_corpus(out_path)) == 2


def test_load_corpus_skips_blank_lines(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("पहिलो वाक्य यहाँ छ\n\nदोस्रो वाक्य यहाँ छ\n", encoding="utf-8")
    assert load_corpus(str(path)) == ["पहिलो वाक्य यहाँ छ", "दोस्रो वाक्य यहाँ छ"]


def test_manifest_json_is_valid(tmp_path):
    src = _fixture(tmp_path, ["नेपाल सरकारले आज बजेट सार्वजनिक गरेको छ"])
    out_dir = str(tmp_path / "out2")
    fc.fetch(out_dir, local=src, limit=5, seed=2)
    with open(os.path.join(out_dir, fc.MANIFEST_NAME), encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["kind"] == "deva_corpus"
    assert manifest["filters"]["min_deva_share"] == fc.MIN_DEVA_SHARE
