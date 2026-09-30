"""
test_lexicon_fetch.py — offline tests for scripts/fetch_nepali_lexicon.py (W-A).

The network path is never exercised: parsing, normalization, hashing and the
manifest are tested through local sources (the air-gapped CLI path).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import fetch_nepali_lexicon as fl  # noqa: E402


def test_parse_wordlist_tolerates_counts_and_comments():
    text = "# comment\nनेपाल 12\n\nकुल\n"
    assert fl.parse_wordlist(text) == ["नेपाल", "कुल"]


def test_parse_sabdakosh_gzip_and_shapes():
    payload = [{"word": "नेपाल", "definitions": []}, {"word": "abc"}]
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    assert fl.parse_sabdakosh(raw) == ["नेपाल", "abc"]
    assert fl.parse_sabdakosh(json.dumps(payload).encode("utf-8")) == \
        ["नेपाल", "abc"]
    assert fl.parse_sabdakosh(json.dumps({"words": ["कुल"]}).encode()) == \
        ["कुल"]


def test_normalize_words_filters_and_deduplicates():
    kept, dropped = fl.normalize_words(["नेपाल", "abc", "क", "१२", "नेपाल"])
    assert kept == {"नेपाल"}
    assert dropped == 3, "invalid entries drop; duplicates are deduplicated"


def test_build_merges_sources_with_stats():
    wordlist = "नेपाल 5\nकुल\n"
    sabdakosh = json.dumps([{"word": "जम्मा"}, {"word": "abc"}]).encode()
    result = fl.build(wordlist, sabdakosh)
    assert result["words"] == sorted({"नेपाल", "कुल", "जम्मा"})
    assert result["stats"]["wordlist_kept"] == 2
    assert result["stats"]["sabdakosh_kept"] == 1
    assert result["stats"]["overlap"] == 0


def test_write_lexicon_header_and_hash(tmp_path):
    path = str(tmp_path / "lex.txt")
    sha = fl.write_lexicon(["कुल", "नेपाल"], path)
    text = open(path, encoding="utf-8").read()
    assert text.startswith("# VeriScript Nepali lexicon v1")
    assert sha == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert text.rstrip().splitlines()[-2:] == ["कुल", "नेपाल"]


def test_fetch_with_local_sources_writes_manifest(tmp_path):
    wordlist = tmp_path / "nep.wordlist"
    wordlist.write_text("नेपाल\n", encoding="utf-8")
    sabdakosh = tmp_path / "sabdakosh.json.gz"
    sabdakosh.write_bytes(gzip.compress(
        json.dumps([{"word": "कुल"}]).encode("utf-8")))

    out_dir = str(tmp_path / "out")
    manifest = fl.fetch(out_dir, str(wordlist), str(sabdakosh))

    assert manifest["kind"] == "nepali_lexicon"
    assert manifest["output"]["words"] == 2
    out = os.path.join(out_dir, manifest["output"]["file"])
    assert os.path.exists(out)
    assert manifest["output"]["sha256"] == hashlib.sha256(
        open(out, "rb").read()).hexdigest()
    assert manifest["sources"][0]["sha256"] == hashlib.sha256(
        wordlist.read_bytes()).hexdigest()
    assert manifest["sources"][0]["license"] == "Apache-2.0"
    assert manifest["sources"][1]["license"] == "MIT"
    assert os.path.exists(os.path.join(out_dir, fl.MANIFEST_NAME))
