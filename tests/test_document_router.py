"""
test_document_router.py — mixed-page router: restored text, untouched non-text (P5).

Text regions come from OCR boxes (measured area coverage 0.975 on letterpress);
everything else is composited back from the original, so logos/photos/signature
cannot be damaged or invented by the restore step.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from document_ocr import Token  # noqa: E402
from document_router import (composite_regions, route_page,  # noqa: E402
                             text_mask_from_tokens)


def _tok(text, bbox):
    return Token(text=text, conf=95, bbox=bbox, granularity="line")


def test_text_mask_covers_boxes_and_leaves_background():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = text_mask_from_tokens(img.shape, [_tok("x", (50, 60, 150, 90))])
    assert mask[75, 100] == 255, "token center must be inside the mask"
    assert mask[10, 10] == 0, "far background must stay outside"
    assert mask.shape == img.shape[:2]


def test_composite_uses_restored_inside_original_outside():
    original = np.zeros((120, 120, 3), dtype=np.uint8)
    restored = np.full((120, 120, 3), 255, dtype=np.uint8)
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[40:80, 40:80] = 255
    out = composite_regions(original, restored, mask, feather=0)
    assert out[60, 60].min() > 250, "text region must be the restored image"
    assert out[5, 5].max() < 5, "non-text region must be the original image"


def test_route_page_keeps_non_text_untouched():
    original = np.zeros((120, 120, 3), dtype=np.uint8)
    original[:, :] = (0, 0, 200)  # red-ish page (logo/photo stand-in)
    restored = np.full((120, 120, 3), 255, dtype=np.uint8)
    toks = [_tok("x", (40, 40, 80, 60))]
    out = route_page(original, restored, toks)
    assert out[5, 5, 2] > 150, "non-text pixel must come from the original"
    assert out[50, 60].min() > 200, "text pixel must come from the restored"


def test_route_page_photo_upscaler_only_touches_photo_regions():
    original = np.zeros((120, 120, 3), dtype=np.uint8)
    restored = np.zeros((120, 120, 3), dtype=np.uint8)
    regions = [{"kind": "photo", "bbox": [10, 10, 50, 50]},
               {"kind": "logo", "bbox": [70, 70, 100, 100]}]
    calls = []

    def upscaler(patch):
        calls.append(patch.shape)
        return np.full_like(patch, 99)

    out = route_page(original, restored, [], regions=regions,
                     photo_upscaler=upscaler)
    assert len(calls) == 1, "upscaler must only see the photo region"
    assert out[30, 30].max() == 99
    assert out[85, 85].max() == 0, "logo must stay original (no upscaler)"


@pytest.mark.skipif("rapidocr" not in __import__("document_ocr").available_backends(),
                    reason="rapidocr unavailable")
def test_pipeline_mixed_router_keeps_non_text(tmp_path):
    import doc_data
    from document_pipeline import run_document_pipeline

    page, _gt = doc_data.render_synthetic_invoice(seed=401, dpi=150)
    res = run_document_pipeline(page, backend="rapidocr", mixed_router=True,
                                out_dir=str(tmp_path), stem="routed")
    assert res.meta["mixed_router"] is True
    assert res.display_bgr.shape == page.shape
    corner = res.display_bgr[:40, :40].astype(int)
    orig = page[:40, :40].astype(int)
    assert np.abs(corner - orig).mean() < 3, \
        "non-text corner must stay the original pixels"


def test_route_page_shapes_and_dtype():
    original = np.full((60, 90, 3), 10, dtype=np.uint8)
    restored = np.full((60, 90, 3), 200, dtype=np.uint8)
    out = route_page(original, restored, [_tok("x", (10, 10, 50, 30))])
    assert out.shape == original.shape and out.dtype == np.uint8
