"""
test_doc_data_mixed.py — synthetic mixed-page fixtures (P5).

A mixed page carries text + logo + photo + signature with exact text GT and
exact region boxes (photo content needs no GT). The router's pre-registered
gates are measured on this set.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_data import (build_mixed_dataset, load_dataset,  # noqa: E402
                      render_mixed_document)


def test_render_mixed_document_has_all_region_kinds():
    img, gt, regions = render_mixed_document(seed=1, dpi=150)
    assert img.ndim == 3 and img.shape[2] == 3
    kinds = {r["kind"] for r in regions}
    assert kinds == {"text", "logo", "photo", "signature"}
    h, w = img.shape[:2]
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    assert "INVOICE" in gt.upper()
    assert "TOTAL" in gt.upper()


def test_render_mixed_document_is_deterministic():
    a = render_mixed_document(seed=2, dpi=150)
    b = render_mixed_document(seed=2, dpi=150)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]
    assert a[2] == b[2]


def test_photo_region_has_texture_variance():
    img, _gt, regions = render_mixed_document(seed=3, dpi=150)
    photo = next(r for r in regions if r["kind"] == "photo")
    x0, y0, x1, y1 = photo["bbox"]
    patch = img[y0:y1, x0:x1]
    assert float(patch.std()) > 5.0, "photo region must not be a flat fill"


def test_build_mixed_dataset_roundtrip(tmp_path):
    manifest = build_mixed_dataset(str(tmp_path / "mixed"), n=2, dpi=150,
                                   seed=7)
    assert manifest["kind"] == "mixed_pages"
    assert len(manifest["entries"]) == 2
    loaded = load_dataset(str(tmp_path / "mixed"))
    for e in loaded["entries"]:
        assert os.path.exists(e["_clean_path"])
        assert os.path.exists(e["_gt_path"])
        assert os.path.exists(os.path.join(str(tmp_path / "mixed"),
                                           e["regions"]))
    kinds = {r["kind"] for r in loaded["entries"][0]["_regions"]}
    assert kinds == {"text", "logo", "photo", "signature"}
