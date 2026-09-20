"""
test_document_eval_smoke.py — unit + smoke tests for the document restore stack.

Fast by default: 2 synthetic pages, RapidOCR only (Tesseract optional/skip).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from degradation_document import degrade_page  # noqa: E402
from document_ocr import Token, available_backends, ocr_page  # noqa: E402


def test_degradation_deterministic():
    page, _gt = doc_data.render_synthetic_invoice(seed=5)
    a = degrade_page(page, seed=11, level="medium")
    b = degrade_page(page, seed=11, level="medium")
    c = degrade_page(page, seed=12, level="medium")
    assert a.shape == page.shape and a.dtype == np.uint8
    assert (a == b).all(), "same seed must be deterministic"
    assert not (a == c).all(), "different seed should differ"


def test_synthetic_invoice_gt_contains_amounts():
    _img, gt = doc_data.render_synthetic_invoice(seed=3)
    assert "INVOICE" in gt
    assert "TOTAL" in gt
    assert any(ch.isdigit() for ch in gt)
    assert gt == doc_data.normalize_text(gt)


def test_metrics_known_values():
    gt = "INVOICE 1200.00 Total Amount"
    hyp = "INVOICE 1200.00 Total Am0unt"
    assert 0 < doc_metrics.cer(gt, hyp) < 0.1
    toks = [Token(text="INVOICE", conf=99, bbox=(0, 0, 1, 1), granularity="word"),
            Token(text="1200.00", conf=50, bbox=(0, 0, 1, 1), granularity="word",
                  flags=["low_conf"]),
            Token(text="Am0unt", conf=90, bbox=(0, 0, 1, 1), granularity="word")]
    stats = doc_metrics.token_stats(toks, gt)
    assert stats.errors == 1
    assert stats.coverage == 0.0  # the flagged token was correct
    assert stats.false_alarm_rate == 1.0
    hall = doc_metrics.hallucination_report(gt, hyp, "INVOICE 1200.00 Total 4m0unt")
    assert hall["invented"] == 1 and hall["invented_digits"] == 1


def test_pdf_text_layer_roundtrip(tmp_path):
    pdf = os.path.join(tmp_path, "demo.pdf")
    doc_data.make_demo_pdf(pdf)
    pages = list(doc_data.pdf_to_pages(pdf, dpi=150))
    assert len(pages) == 1
    _idx, img, gt = pages[0]
    assert img.shape[0] > 1000 and "INVOICE" in gt and "1424.14" in gt


@pytest.mark.skipif("rapidocr" not in available_backends(), reason="rapidocr unavailable")
def test_ocr_rapidocr_on_degraded_invoice():
    page, gt = doc_data.render_synthetic_invoice(seed=7)
    degraded = degrade_page(page, seed=7, level="medium")
    res = ocr_page(degraded, backend="rapidocr")
    assert len(res.tokens) > 5
    assert res.meta["n_tokens"] == len(res.tokens)
    assert doc_metrics.cer(gt, res.text) < 0.5, "RapidOCR should read most of a medium-degraded invoice"


@pytest.mark.skipif("rapidocr" not in available_backends(), reason="rapidocr unavailable")
def test_eval_document_pipeline(tmp_path):
    from eval_document import run_evaluation
    data_dir = os.path.join(tmp_path, "synthetic")
    manifest = doc_data.build_synthetic_dataset(data_dir, n=2, levels=("mild",))
    entries = manifest["entries"]
    for e in entries:
        e["_gt_path"] = os.path.join(data_dir, e["gt"])
        e["_degraded_path"] = os.path.join(data_dir, e["degraded"])
        e["_clean_path"] = os.path.join(data_dir, e["clean"])
    report = run_evaluation(entries, ["raw"], ["rapidocr"])
    assert report["summary"]["raw@rapidocr"]["pages"] == 2
    assert report["summary"]["raw@rapidocr"]["cer"] is not None
    assert len(report["per_page"]) == 2


def test_tesseract_adapter_optional():
    status = "tesseract" in available_backends()
    from document_ocr import TesseractBackend
    ok, why = TesseractBackend().available()
    assert ok == status
    if not ok:
        assert "winget" in why or "not found" in why
