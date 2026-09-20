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
