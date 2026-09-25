"""
test_harvest_textbooks.py — W-D harvest: DSpace parsers + gated MOEST flow.

Network is never touched: `_get`/`_download`/`probe_pdf` are monkeypatched.
"""
from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import harvest_nepali_textbooks as ht  # noqa: E402

MOEST_SEARCH = (
    '<html><body>'
    '<a href="/handle/123456789/424">teacher guide</a>'
    '<a href="/handle/123456789/579">directive</a>'
    '<a href="/handle/123456789/424">duplicate</a>'
    '</body></html>'
)
MOEST_ITEM = (
    '<html><head>'
    '<meta name="citation_title" content="&#x20;Curriculum&#x20;Guide" />'
    '<meta name="citation_pdf_url" content="http://x/bitstream/1/2/a.pdf" />'
    '</head></html>'
)
CORNELL_SEARCH = {"_embedded": {"searchResult": {"_embedded": {"objects": [
    {"_embedded": {"indexableObject": {
        "uuid": "u1", "handle": "1813/1",
        "metadata": {"dc.title": [{"value": "Mero Nepali"}]}}}},
    {"_embedded": {"indexableObject": {
        "uuid": "u2", "handle": "1813/2",
        "metadata": {"dc.title": [{"value": "Hamro Nepali"}]}}}},
]}}}}
CORNELL_BUNDLES = {"_embedded": {"bundles": [
    {"name": "ORIGINAL", "_embedded": {"bitstreams": [
        {"name": "book.pdf", "sizeBytes": 123,
         "_links": {"content": {"href": "http://x/content"}}},
        {"name": "thumb.jpg", "sizeBytes": 1,
         "_links": {"content": {"href": "http://x/thumb"}}},
    ]}},
    {"name": "TEXT", "_embedded": {"bitstreams": []}},
]}}


def test_parse_moest_search_dedupes_and_keeps_order():
    assert ht.parse_moest_search(MOEST_SEARCH) == [
        "/handle/123456789/424", "/handle/123456789/579"]


def test_parse_moest_item_unescapes_metadata():
    info = ht.parse_moest_item(MOEST_ITEM)
    assert info["title"] == "Curriculum Guide"
    assert info["pdf_url"] == "http://x/bitstream/1/2/a.pdf"
    assert ht.parse_moest_item("<html></html>") == {"title": "", "pdf_url": ""}


def test_parse_cornell_items():
    items = ht.parse_cornell_items(CORNELL_SEARCH)
    assert [(i["uuid"], i["handle"], i["title"]) for i in items] == [
        ("u1", "1813/1", "Mero Nepali"), ("u2", "1813/2", "Hamro Nepali")]


def test_pick_cornell_pdf_prefers_original_pdf():
    pdf = ht.pick_cornell_pdf(CORNELL_BUNDLES)
    assert pdf == {"name": "book.pdf", "bytes": 123, "url": "http://x/content"}
    nested = {"_embedded": {"bundles": [
        {"name": "ORIGINAL", "_embedded": {"bitstreams": {
            "_embedded": {"bitstreams": [
                {"name": "nested.pdf", "sizeBytes": 9,
                 "_links": {"content": {"href": "http://x/nested"}}}]}}}}]}}
    assert ht.pick_cornell_pdf(nested)["url"] == "http://x/nested"
    assert ht.pick_cornell_pdf({"_embedded": {"bundles": []}}) is None
    assert ht.pick_cornell_pdf(
        {"_embedded": {"bundles": [{"name": "ORIGINAL",
                                    "_embedded": {"bitstreams": []}}]}}) is None


def test_harvest_moest_gates_and_removes_rejected(tmp_path, monkeypatch):
    def fake_get(url, timeout=90):
        if "simple-search" in url:
            return MOEST_SEARCH
        return MOEST_ITEM

    def fake_download(url, dst, retries=3):
        with open(dst, "wb") as f:
            f.write(b"%PDF-1.4 fake")

    def fake_probe(path, max_pages=8):
        return {"pages": 10, "chars_per_page": 800.0, "devanagari_ratio": 0.9,
                "invalid_token_rate": 0.0}

    monkeypatch.setattr(ht, "_get", fake_get)
    monkeypatch.setattr(ht, "_download", fake_download)
    monkeypatch.setattr(ht, "probe_pdf", fake_probe)
    monkeypatch.setattr(ht, "gate_reason", lambda info: (
        "mojibake" if info["devanagari_ratio"] < 0.5 else None))
    report = ht.harvest_moest(str(tmp_path), limit=2,
                              queries=["textbook"])
    assert len(report["accepted"]) == 2
    assert report["accepted"][0]["title"] == "Curriculum Guide"
    assert report["rejected"] == []
    assert all(os.path.exists(os.path.join(str(tmp_path), r["file"]))
               for r in report["accepted"])


def test_harvest_moest_rejects_legacy_fonts(tmp_path, monkeypatch):
    def fake_get(url, timeout=90):
        return MOEST_SEARCH if "simple-search" in url else MOEST_ITEM

    def fake_download(url, dst, retries=3):
        with open(dst, "wb") as f:
            f.write(b"%PDF-1.4 fake")

    monkeypatch.setattr(ht, "_get", fake_get)
    monkeypatch.setattr(ht, "_download", fake_download)
    monkeypatch.setattr(ht, "probe_pdf", lambda path, max_pages=8: {
        "pages": 87, "chars_per_page": 749.0, "devanagari_ratio": 0.02,
        "invalid_token_rate": 0.083})
    monkeypatch.setattr(ht, "gate_reason",
                        lambda info: "mojibake/non-Unicode text")
    report = ht.harvest_moest(str(tmp_path), limit=1, queries=["textbook"])
    assert report["accepted"] == []
    rec = report["rejected"][0]
    assert rec["verdict"] == "rejected" and "mojibake" in rec["reason"]
    assert not os.path.exists(os.path.join(str(tmp_path), rec["file"]))


def test_harvest_cornell_downloads_first_pdf(tmp_path, monkeypatch):
    def fake_get(url, timeout=90):
        return json.dumps(CORNELL_BUNDLES if "/bundles" in url
                          else CORNELL_SEARCH)

    def fake_download(url, dst, retries=3):
        with open(dst, "wb") as f:
            f.write(b"%PDF-1.4 fake scan")

    monkeypatch.setattr(ht, "_get", fake_get)
    monkeypatch.setattr(ht, "_download", fake_download)
    report = ht.harvest_cornell(str(tmp_path), limit=2)
    assert len(report["downloaded"]) == 2
    assert report["downloaded"][0]["handle"] == "1813/1"
    assert report["downloaded"][0]["title"] == "Mero Nepali"
    assert report["errors"] == []
