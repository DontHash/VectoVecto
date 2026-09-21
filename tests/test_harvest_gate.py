"""
test_harvest_gate.py — harvest gate v2: GT integrity + record reasons (N2).

The v1 gate only checked chars/page and Devanagari ratio. v2 also rejects text
layers whose Devanagari contains invalid combining sequences (>2% of tokens),
and records the rejection reason for every file.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import harvest_nepali_pdfs as hp  # noqa: E402


def _info(**kw):
    base = {"pages": 10, "sampled_pages": 8, "chars": 8000,
            "chars_per_page": 1000.0, "devanagari_ratio": 0.9,
            "invalid_token_rate": 0.0}
    base.update(kw)
    return base


def test_gate_accepts_clean_document():
    assert hp.gate_reason(_info()) is None


def test_gate_rejects_invalid_sequences():
    reason = hp.gate_reason(_info(invalid_token_rate=0.05))
    assert reason is not None and "invalid" in reason


def test_gate_rejects_scan_and_mojibake():
    assert "chars" in hp.gate_reason(_info(chars_per_page=12.0))
    assert "deva" in hp.gate_reason(_info(devanagari_ratio=0.1))


def test_gate_boundary_is_strict():
    assert hp.gate_reason(_info(invalid_token_rate=hp.MAX_INVALID_RATE)) is None
    assert hp.gate_reason(_info(invalid_token_rate=hp.MAX_INVALID_RATE + 0.001))


def test_harvest_records_reason_and_removes_rejected(tmp_path, monkeypatch):
    seen = {}

    def fake_download(url, dst, retries=3):
        seen["dst"] = dst
        with open(dst, "wb") as f:
            f.write(b"%PDF-1.4 fake")

    monkeypatch.setattr(hp, "_download", fake_download)
    monkeypatch.setattr(hp, "probe_pdf", lambda path, max_pages=8: _info(
        invalid_token_rate=0.5))
    report = hp.harvest(str(tmp_path), urls=["https://example.org/bad.pdf"])
    assert report["accepted"] == []
    rec = report["rejected"][0]
    assert "invalid" in rec["reason"]
    assert not os.path.exists(seen["dst"])


def test_harvest_accepts_and_keeps_clean(tmp_path, monkeypatch):
    def fake_download(url, dst, retries=3):
        with open(dst, "wb") as f:
            f.write(b"%PDF-1.4 fake")

    monkeypatch.setattr(hp, "_download", fake_download)
    monkeypatch.setattr(hp, "probe_pdf", lambda path, max_pages=8: _info())
    report = hp.harvest(str(tmp_path), urls=["https://example.org/good.pdf"])
    assert len(report["accepted"]) == 1
    assert report["accepted"][0]["reason"] is None


def test_probe_pdf_reports_invalid_rate(tmp_path):
    import doc_data

    pdf = str(tmp_path / "demo.pdf")
    doc_data.make_demo_pdf(pdf)
    info = hp.probe_pdf(pdf)
    assert info["invalid_token_rate"] == 0.0
    assert info["devanagari_ratio"] == 0.0


def test_probe_pdf_releases_file_lock(tmp_path):
    import doc_data

    pdf = str(tmp_path / "demo.pdf")
    doc_data.make_demo_pdf(pdf)
    hp.probe_pdf(pdf)
    os.remove(pdf)  # pdfium must not hold the handle (rejected files get deleted)
