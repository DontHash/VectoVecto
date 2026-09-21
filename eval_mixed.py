"""
eval_mixed.py — mixed-page router gates on the synthetic mixed_pages set (P5).

Pre-registered gates (fixed before measuring; auto-enable only if all pass,
else the router stays opt-in via `--mixed-router`):

  1. text   : CER on the routed page <= plain-page CER * 1.02 (relative +2%)
  2. non-text: routed PSNR/SSIM >= the naive full-page-restore baseline
               ("Lanczos" = no enhancement: keeping the original wins by
               construction; the measurement records the harm the router avoids)
  3. invented: no token in the routed OCR that is absent from both the plain
               OCR and the GT
  4. cost   : router overhead <= +1 s/page

Usage:
    python eval_mixed.py --data-dir data/doc_eval/mixed_pages \
        --json out/mixed_gates.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import ocr_page  # noqa: E402
from document_restore import fit_max_side, restore_document  # noqa: E402
from document_router import route_page, text_mask_from_tokens  # noqa: E402

TEXT_CER_SLACK = 1.02
COST_BUDGET_S = 1.0


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """PSNR over masked pixels only (uint8 images, 0/255 mask); inf if equal."""
    m = mask > 0
    if not m.any():
        return float("inf")
    diff = a[m].astype(np.float64) - b[m].astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean SSIM over masked pixels (channel-aware)."""
    from skimage.metrics import structural_similarity
    _score, full = structural_similarity(a, b, channel_axis=2, data_range=255,
                                         full=True)
    m = mask > 0
    if not m.any():
        return 1.0
    if full.ndim == 3:
        return float(full[m].mean())
    return float(full[m].mean())


def gate_verdict(cer_plain: float, cer_routed: float, psnr_routed: float,
                 psnr_baseline: float, invented: int, cost_s: float) -> Dict:
    """Evaluate the four pre-registered gates."""
    v = {
        "text": cer_routed <= cer_plain * TEXT_CER_SLACK + 1e-9,
        "non_text": psnr_routed + 1e-9 >= psnr_baseline,
        "invented": invented == 0,
        "cost": cost_s <= COST_BUDGET_S,
    }
    v["pass"] = all(v.values())
    return v


def run(data_dir: str, lang: str = "en", limit: int = 0) -> Dict:
    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"]
    if limit:
        entries = entries[:limit]
    rows: List[Dict] = []
    for i, e in enumerate(entries, 1):
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        gt = open(e["_gt_path"], encoding="utf-8").read()
        img, fit = fit_max_side(img)  # same coordinate space as the pipeline
        regions = e.get("_regions", [])
        if fit < 1.0:
            regions = [{"kind": r["kind"],
                        "bbox": [int(v * fit) for v in r["bbox"]]}
                       for r in regions]

        plain = ocr_page(img, backend="rapidocr", lang=lang)
        cer_plain = doc_metrics.cer(gt, plain.text)

        restored = restore_document(img)["display_bgr"]
        t0 = time.time()
        routed = route_page(img, restored, plain.tokens, regions=regions)
        route_s = time.time() - t0

        routed_ocr = ocr_page(routed, backend="rapidocr", lang=lang)
        cer_routed = doc_metrics.cer(gt, routed_ocr.text)

        mask = text_mask_from_tokens(img.shape, plain.tokens)
        non_text = np.where(mask > 0, 0, 255).astype(np.uint8)
        psnr_routed = masked_psnr(img, routed, non_text)
        psnr_naive = masked_psnr(img, restored, non_text)
        ssim_routed = masked_ssim(img, routed, non_text)
        ssim_naive = masked_ssim(img, restored, non_text)

        gt_set = {doc_metrics._norm_tok(t) for t in doc_metrics.tokens_of(gt)}
        plain_set = {doc_metrics._norm_tok(t)
                     for t in doc_metrics.tokens_of(plain.text)}
        invented = sum(1 for t in doc_metrics.tokens_of(routed_ocr.text)
                       if doc_metrics._norm_tok(t)
                       and doc_metrics._norm_tok(t) not in gt_set
                       and doc_metrics._norm_tok(t) not in plain_set)

        row = {
            "page": e["id"], "cer_plain": cer_plain, "cer_routed": cer_routed,
            "psnr_routed": psnr_routed, "psnr_naive": psnr_naive,
            "ssim_routed": ssim_routed, "ssim_naive": ssim_naive,
            "invented": invented, "route_s": route_s,
        }
        row["gates"] = gate_verdict(cer_plain, cer_routed, psnr_routed,
                                    psnr_naive, invented, route_s)
        rows.append(row)
        print(f"  [{i}/{len(entries)}] {e['id']} CER {cer_plain:.3f}->"
              f"{cer_routed:.3f} PSNR(nt) "
              f"{psnr_routed if np.isinf(psnr_routed) else round(psnr_routed, 1)}"
              f" vs {round(psnr_naive, 1)} invented {invented}", flush=True)

    def _mean(key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)) if vals else None

    summary = {
        "pages": len(rows),
        "cer_plain": _mean("cer_plain"),
        "cer_routed": _mean("cer_routed"),
        "psnr_naive_non_text": _mean("psnr_naive"),
        "ssim_naive_non_text": _mean("ssim_naive"),
        "invented_tokens": int(sum(r["invented"] for r in rows)),
        "route_s_per_page": _mean("route_s"),
        "gates": {
            "text": all(r["gates"]["text"] for r in rows) if rows else None,
            "non_text": all(r["gates"]["non_text"] for r in rows) if rows else None,
            "invented": all(r["gates"]["invented"] for r in rows) if rows else None,
            "cost": all(r["gates"]["cost"] for r in rows) if rows else None,
        },
    }
    summary["gates"]["pass"] = all(summary["gates"][k] for k in
                                   ("text", "non_text", "invented", "cost"))
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Mixed-page router gates")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    result = run(args.data_dir, lang=args.lang, limit=args.limit)
    s = result["summary"]
    print("\n=== MIXED ROUTER GATES (pre-registered) ===")
    for k, v in s.items():
        print(f"  {k:<24} {v}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[mixed] wrote {args.json}")


if __name__ == "__main__":
    main()
