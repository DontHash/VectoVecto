"""
test_cli_document.py — CLI document mode end-to-end (in-process, no subprocess).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from degradation_document import degrade_page  # noqa: E402
from document_ocr import available_backends  # noqa: E402


def _doc_args(**overrides) -> argparse.Namespace:
    base = dict(
        mode="document", input=None, output=None, model="auto", scale=4,
        recursive=False, flat=False, skip_existing=False, suffix="", format="png",
        quality=95, device="auto", tile=None, tta=False, no_fp16=False, grain=0.0,
        report=None, workers=1, limit=0, ocr=None, lang=None, deskew=False,
        repass_digits=False, max_pages=1, dpi=0, no_pdf=False, no_overlay=False,
        no_txt=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.skipif(not available_backends(), reason="no OCR backend available")
def test_cli_document_mode_writes_outputs(tmp_path):
    from cli import run_document_mode

    page, _gt = doc_data.render_synthetic_invoice(seed=51, dpi=150)
    degraded = degrade_page(page, seed=5101, level="mild")
    src = tmp_path / "scan.png"
    cv2.imwrite(str(src), degraded)
    out_dir = tmp_path / "out"
    report = tmp_path / "report.json"

    args = _doc_args(input=str(src), output=str(out_dir), report=str(report))
    rc = run_document_mode(args)
    assert rc == 0

    written = sorted(os.listdir(out_dir))
    assert "scan.pdf" in written
    assert "scan_overlay.png" in written
    assert "scan.txt" in written
    assert "scan.json" in written

    payload = json.load(open(report, encoding="utf-8"))
    assert payload["summary"]["ok"] == 1
    assert payload["records"][0]["status"] == "ok"

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(out_dir / "scan.pdf"))
    text = doc[0].get_textpage().get_text_range().upper()
    assert "INVOICE" in text


def _make_pdf(path, n_pages=3):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for i in range(n_pages):
        c.setFont("Helvetica-Bold", 22)
        c.drawString(72, 780, f"INVOICE PAGE {i}")
        c.setFont("Helvetica", 12)
        c.drawString(72, 750, f"TOTAL {1200 + i}.00")
        c.showPage()
    c.save()


@pytest.mark.skipif(not available_backends(), reason="no OCR backend available")
def test_cli_document_all_pages_writes_combined(tmp_path):
    from cli import run_document_mode

    src = tmp_path / "multi.pdf"
    _make_pdf(src, 3)
    out_dir = tmp_path / "out"
    args = _doc_args(input=str(src), output=str(out_dir), max_pages=0, dpi=100,
                     report=str(tmp_path / "report.json"))
    assert run_document_mode(args) == 0

    combined = out_dir / "multi_combined.pdf"
    assert combined.exists(), "max-pages 0 must write the combined searchable PDF"
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(combined))
    assert len(doc) == 3
    for i in range(3):
        text = doc[i].get_textpage().get_text_range().upper()
        assert f"PAGE {i}" in text, f"searchable text missing on page {i}"

    txt = (out_dir / "multi_combined.txt").read_text(encoding="utf-8").upper()
    assert "PAGE 0" in txt and "PAGE 2" in txt
    assert (out_dir / "multi_p000.pdf").exists(), "per-page artifacts stay"


@pytest.mark.skipif(not available_backends(), reason="no OCR backend available")
def test_cli_document_mode_skip_existing(tmp_path):
    from cli import run_document_mode

    page, _gt = doc_data.render_synthetic_invoice(seed=52, dpi=150)
    src = tmp_path / "scan.png"
    cv2.imwrite(str(src), degrade_page(page, seed=5201, level="mild"))
    out_dir = tmp_path / "out"

    first = _doc_args(input=str(src), output=str(out_dir))
    assert run_document_mode(first) == 0
    before = os.path.getmtime(out_dir / "scan.pdf")

    second = _doc_args(input=str(src), output=str(out_dir), skip_existing=True,
                       report=str(tmp_path / "r2.json"))
    assert run_document_mode(second) == 0
    assert os.path.getmtime(out_dir / "scan.pdf") == before, "existing output must be skipped"
    payload = json.load(open(tmp_path / "r2.json", encoding="utf-8"))
    assert payload["summary"]["skipped"] == 1
