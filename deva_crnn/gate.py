"""
gate.py — W1 adopt-if gate: line-level digit-exact + bagCER on frozen sets.

Gate (pre-registered): digit-exact >= 0.75 on digit-bearing lines, bagCER not
worse than the RapidOCR line baseline, <= +1 s/page. Sets:
  * heiDATA frozen pages -> ALTO line crops (human GT)
  * nepali_pdf_v2 anchor pages -> RapidOCR line boxes, Gemini anchor GT
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import normalize_line  # noqa: E402
from doc_metrics import cer, cer_bag, digit_string, digit_tokens  # noqa: E402

DIGIT_EXACT_BAR = 0.75


def line_stats(gts: List[str], hyps: List[str]) -> Dict:
    """Digit-exact rate over digit-bearing lines + mean CER/bagCER."""
    pairs = [(g, h) for g, h in zip(gts, hyps) if g.strip()]
    digit_pairs = [(g, h) for g, h in pairs if digit_tokens(g)]
    exact = [1.0 if digit_string(g) == digit_string(h) else 0.0
             for g, h in digit_pairs]
    return {
        "lines": len(pairs),
        "digit_lines": len(digit_pairs),
        "digit_exact": float(np.mean(exact)) if exact else None,
        "cer": float(np.mean([cer(g, h) for g, h in pairs])) if pairs else None,
        "bagcer": (float(np.mean([cer_bag(g, h) for g, h in pairs]))
                   if pairs else None),
    }


def line_values(gts: List[str], hyps: List[str]) -> Dict[str, List[float]]:
    """Per-line values for bootstrap CIs (digit-exact 0/1 + CER)."""
    pairs = [(g, h) for g, h in zip(gts, hyps) if g.strip()]
    digit_pairs = [(g, h) for g, h in pairs if digit_tokens(g)]
    return {
        "digit_exact": [1.0 if digit_string(g) == digit_string(h) else 0.0
                        for g, h in digit_pairs],
        "cer": [cer(g, h) for g, h in pairs],
        "bagcer": [cer_bag(g, h) for g, h in pairs],
    }


def heidata_line_crops(data_dir: str, limit: int = 0) -> Tuple[List[str], List[np.ndarray]]:
    """(texts, crops) from frozen heiDATA pages + ALTO line boxes."""
    import doc_data
    manifest = doc_data.load_dataset(data_dir)
    texts: List[str] = []
    crops: List[np.ndarray] = []
    for e in manifest["entries"]:
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        boxes = json.load(open(e["_boxes_path"], encoding="utf-8"))
        for b in boxes:
            x0, y0, x1, y1 = b["bbox"]
            pad = 4
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            if crop.size == 0:
                continue
            texts.append(b["text"])
            crops.append(crop)
        if limit and len(crops) >= limit:
            break
    return texts, crops


def v2_anchor_pages(data_dir: str, readings_path: str,
                    limit: int = 0) -> List[Dict]:
    """Per anchor page: RapidOCR line boxes (image + crops) + Gemini GT."""
    import doc_data
    from document_ocr import ocr_page
    manifest = doc_data.load_dataset(data_dir)
    readings = {r["page"]: r["readings"]["gemini"]
                for r in json.load(open(readings_path, encoding="utf-8"))["rows"]}
    pages: List[Dict] = []
    for e in manifest["entries"]:
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None or e["id"] not in readings:
            continue
        rec = ocr_page(img, backend="rapidocr", lang="ne")
        crops = []
        for t in rec.tokens:
            x0, y0, x1, y1 = t.bbox
            crops.append(img[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3])
        pages.append({"page": e["id"], "gt": readings[e["id"]],
                      "boxes": [t.bbox for t in rec.tokens], "crops": crops})
        if limit and len(pages) >= limit:
            break
    return pages
