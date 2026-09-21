"""
test_doc_data_gt_validity.py — GT audit recorded in pdf dataset manifests (N3).

`build_pdf_dataset(gt_validity=True)` stores the Devanagari invalid-sequence
rate per page entry, so a freeze manifest carries the GT-quality record with
the data it describes.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_data import build_pdf_dataset, make_demo_pdf  # noqa: E402


def _demo(tmp_path):
    pdf = str(tmp_path / "demo.pdf")
    make_demo_pdf(pdf)
    return pdf


def test_build_pdf_dataset_records_gt_validity(tmp_path):
    m = build_pdf_dataset([_demo(tmp_path)], str(tmp_path / "ds"), dpi=72,
                          degrade=False, gt_validity=True)
    e = m["entries"][0]
    assert e["gt_invalid_tokens"] == 0
    assert e["gt_invalid_token_rate"] == 0.0


def test_build_pdf_dataset_default_has_no_audit_fields(tmp_path):
    m = build_pdf_dataset([_demo(tmp_path)], str(tmp_path / "ds"), dpi=72,
                          degrade=False)
    assert "gt_invalid_token_rate" not in m["entries"][0]
