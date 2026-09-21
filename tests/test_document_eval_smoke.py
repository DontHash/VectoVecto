"""
test_document_eval_smoke.py — unit + smoke tests for the document restore stack.

Fast by default: 2 synthetic pages, RapidOCR only (Tesseract optional/skip).
"""
from __future__ import annotations

import json
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


def test_bag_cer_is_order_insensitive():
    gt = "TOTAL 9.00 TAX 1.20"
    reordered = "TAX 1.20 TOTAL 9.00"
    assert doc_metrics.cer(gt, reordered) > 0.4
    assert doc_metrics.cer_bag(gt, reordered) == 0.0


def test_hallucination_peer_refs_ignore_corroborated_text():
    gt = "TOTAL 9.00"
    raw = "TOTAL 9.00"
    restored = "TOTAL 9.00 CASHIER Ravi"
    strict = doc_metrics.hallucination_report(gt, raw, restored)
    assert strict["invented"] == 2, "synthetic/strict mode counts every unseen token"
    lenient = doc_metrics.hallucination_report(gt, raw, restored,
                                               extra_refs=["CASHIER 500.00"])
    assert lenient["invented"] == 1, "peer-confirmed CASHIER is not an invention"


def test_digit_cer_isolates_money_errors():
    gt = "Subtotal 1200.00 VAT 156.00 TOTAL 1356.00"
    same_words_bad_digits = "Subtotal 1200.08 VAT 156.00 TOTAL 1356.00"
    assert doc_metrics.digit_cer(gt, same_words_bad_digits) > 0.0
    assert doc_metrics.digit_cer(gt, gt) == 0.0
    assert doc_metrics.digit_cer(gt, "1356.00 156.00 1200.00 TOTAL VAT Subtotal",
                                 bag=True) == 0.0

    no_gt_digits = doc_metrics.digit_cer("TOTAL DUE", "TOTAL DUE")
    assert no_gt_digits == 0.0
    assert doc_metrics.digit_cer("TOTAL DUE", "TOTAL 42") == 1.0


def test_digit_coverage_money_metric():
    gt = "TOTAL 1200.00 TAX 156.00"
    toks = [Token(text="TOTAL", conf=99, bbox=(0, 0, 1, 1), granularity="word"),
            Token(text="1200.08", conf=50, bbox=(0, 0, 1, 1), granularity="word",
                  flags=["digit_conflict"]),
            Token(text="TAX", conf=99, bbox=(0, 0, 1, 1), granularity="word"),
            Token(text="156.00", conf=99, bbox=(0, 0, 1, 1), granularity="word")]
    stats = doc_metrics.token_stats(toks, gt)
    assert stats.digit_errors == 1
    assert stats.digit_coverage == 1.0, "the wrong amount was flagged"
    assert stats.digit_false_alarm_rate == 0.0

    toks[3].flags.append("digit_conflict")
    stats2 = doc_metrics.token_stats(toks, gt)
    assert stats2.digit_false_alarm_rate == 0.5, "one flagged amount was actually right"


def test_hf_gt_adapters():
    sroie = {"words": ["TAN WOON YANN", "DOCUMENT NO : TD01167104", "TOTAL 9.00"]}
    gt = doc_data.gt_from_sroie(sroie)
    assert gt.startswith("TAN WOON YANN") and "TOTAL 9.00" in gt

    cord = {"ground_truth": json.dumps({"gt_parse": {
        "menu": [{"nm": "Nasi Campur", "price": "75,000"}],
        "total": {"total_price": "75,000"},
    }})}
    gt = doc_data.gt_from_cord(cord)
    assert "Nasi Campur" in gt and "75,000" in gt and "total_price" not in gt

    assert doc_data.gt_from_generic({"text": "hello world"}) == "hello world"
    assert doc_data.gt_from_generic({"nope": 1}) == ""


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
    assert report["summary"]["raw@rapidocr"]["digit_exact_rate"] is not None
    assert len(report["per_page"]) == 2


def test_tesseract_adapter_optional():
    status = "tesseract" in available_backends()
    from document_ocr import TesseractBackend
    ok, why = TesseractBackend().available()
    assert ok == status
    if not ok:
        assert "winget" in why or "not found" in why


def test_dual_stream_gate_flags_digit_conflicts():
    from document_ocr import OCRResult, compare_digit_streams

    primary = OCRResult("", [Token(text="TOTAL 1200.00", conf=95,
                                   bbox=(100, 100, 300, 130), granularity="word",
                                   backend="x")], "x")
    alt = OCRResult("", [Token(text="TOTAL 1280.00", conf=90,
                               bbox=(100, 100, 300, 130), granularity="word",
                               backend="y")], "y")
    assert compare_digit_streams(primary, alt) == 1
    assert primary.tokens[0].alt_text == "TOTAL 1280.00"
    assert "digit_conflict" in primary.tokens[0].flags

    agree = OCRResult("", [Token(text="TOTAL 1200.00", conf=80,
                                 bbox=(100, 100, 300, 130), granularity="word",
                                 backend="y")], "y")
    primary2 = OCRResult("", [Token(text="TOTAL 1200.00", conf=95,
                                    bbox=(100, 100, 300, 130), granularity="word",
                                    backend="x")], "x")
    assert compare_digit_streams(primary2, agree) == 0


def test_recommended_stream_table_covers_backends():
    from document_ocr import RECOMMENDED_STREAM
    assert RECOMMENDED_STREAM.get("rapidocr") == "raw"
    assert RECOMMENDED_STREAM.get("tesseract", "").startswith("restore")


def test_digit_repass_records_evidence_without_polluting_flags():
    from document_ocr import apply_digit_repass

    img = np.full((60, 300, 3), 255, dtype=np.uint8)

    agree = [Token(text="TOTAL 1200.00", conf=90, bbox=(10, 10, 200, 50),
                   granularity="word", backend="x")]
    conflicts = apply_digit_repass(agree, img, lambda crop: ("TOTAL 1200.00", 99.0))
    assert conflicts == 0
    assert agree[0].repass_text == "TOTAL 1200.00"
    assert agree[0].flags == [], "agreement must not add flags (coverage stays honest)"

    disagree = [Token(text="TOTAL 1200.00", conf=90, bbox=(10, 10, 200, 50),
                      granularity="word", backend="x")]
    assert apply_digit_repass(disagree, img, lambda crop: ("TOTAL 1260.00", 95.0)) == 1
    assert "digit_conflict" in disagree[0].flags
    assert disagree[0].alt_text == "TOTAL 1260.00"

    high_conf = [Token(text="TOTAL 999", conf=99, bbox=(10, 10, 200, 50),
                       granularity="word", backend="x")]
    assert apply_digit_repass(high_conf, img, lambda crop: ("WRONG", 10.0)) == 0
    assert high_conf[0].repass_text is None, "high-confidence tokens are not re-read"

    nondigit = [Token(text="TOTAL", conf=50, bbox=(10, 10, 200, 50),
                      granularity="word", backend="x")]
    assert apply_digit_repass(nondigit, img, lambda crop: ("X", 1.0)) == 0
    assert nondigit[0].repass_text is None
