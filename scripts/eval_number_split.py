#!/usr/bin/env python3
"""eval_number_split.py — pre-registered gate for merged number-run splitting
(Appendix AG).

Measures `split_numbers` off/on on a frozen dataset with the exact metric
definitions of the two harnesses the gate names:

  * pipeline page CER / bagCER / digit-exact / invented: the Appendix Y
    definitions (`run_document_pipeline` + `doc_metrics`; repass off, which
    does not change text — Appendix AG measured the 0.2550 baseline with it)
  * digit-token review-queue recall@K / precision@K: the `eval_flags`
    definitions (`ocr_page` with the shipped Devanagari repass, then
    `review_queue`; the shipped pipeline enables it)
  * engagement counters and output identity (pages whose OCR text changed)

Usage:
    python scripts/eval_number_split.py \
        --data-dir data/doc_eval/nepali_pdf_v2 \
        --frozen evals/manifests/nepali_pdf_v2.json --limit 10 \
        --cer-target 0.22 --json out/number_split_hard10.json
    python scripts/eval_number_split.py \
        --data-dir data/doc_eval/heidata_printed \
        --frozen evals/manifests/heidata_printed_v1.json --identity
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.core import data as doc_data  # noqa: E402
from veriscript.core import metrics as doc_metrics  # noqa: E402
from veriscript.document.ocr import ocr_page, review_queue  # noqa: E402

DEVA_DIGITS = {chr(0x0966 + i) for i in range(10)}
LATENCY_BUDGET_S = 1.0


def _digit_tokens(text: str) -> List[str]:
    return [t for t in doc_metrics.tokens_of(text)
            if any(c in DEVA_DIGITS or c.isdigit() for c in t)]


def _load_rows(data_dir: str, limit: int = 0) -> List[Dict]:
    manifest = doc_data.load_dataset(data_dir)
    rows = []
    for e in manifest["entries"]:
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        try:
            gt = open(e["_gt_path"], encoding="utf-8").read()
        except Exception:  # noqa: BLE001
            gt = ""
        rows.append({"page": e["id"], "img": img, "gt": gt})
        if limit and len(rows) >= limit:
            break
    return rows


def _queue_metrics(tokens, gt: str) -> Dict:
    """Digit-token queue hits/queue size (eval_flags definitions)."""
    gt_digits = {doc_metrics.digit_string(t) for t in _digit_tokens(gt)}
    gt_digits.discard("")
    errs, digit_total = [], 0
    for t in tokens:
        if not t.has_digits:
            continue
        digit_total += 1
        d = doc_metrics.digit_string(t.text)
        if d and d not in gt_digits:
            errs.append(t)
    queue = review_queue(tokens)
    out = {"digit_tokens": digit_total, "errors": len(errs),
           "queue": len(queue)}
    for k in (5, 10):
        top = queue[:k]
        out[f"hits@{k}"] = sum(1 for t in top if any(t is e for e in errs))
        out[f"top@{k}"] = len(top)
    return out


def _run_arm(rows: List[Dict], lang: str, split: bool,
             repass_digits: bool = False) -> Dict:
    from veriscript.document.pipeline import run_document_pipeline

    per_page = []
    for r in rows:
        t0 = time.time()
        pres = run_document_pipeline(r["img"], backend="rapidocr", lang=lang,
                                     repass_digits=repass_digits,
                                     split_numbers=split)
        pipeline_s = time.time() - t0
        gt = r["gt"]
        hall = doc_metrics.hallucination_report(gt, "", pres.ocr.text)
        t1 = time.time()
        res = ocr_page(r["img"], backend="rapidocr", lang=lang,
                       recheck_digits=True, split_numbers=split)
        flags_s = time.time() - t1
        q = _queue_metrics(res.tokens, gt)
        per_page.append({
            "page": r["page"],
            "cer": doc_metrics.cer(gt, pres.ocr.text),
            "cer_bag": doc_metrics.cer_bag(gt, pres.ocr.text),
            "digit_exact": 1.0 if doc_metrics.digit_exact(gt, pres.ocr.text) else 0.0,
            "invented": int(hall.get("invented", 0)),
            "invented_digits": int(hall.get("invented_digits", 0)),
            "pipeline_seconds": round(pipeline_s, 3),
            "flags_seconds": round(flags_s, 3),
            "queue": q,
            "split": res.meta.get("number_split") or {},
            "text": res.text,
        })
    errs_total = sum(p["queue"]["errors"] for p in per_page)
    summary = {
        "pages": len(per_page),
        "cer": float(np.mean([p["cer"] for p in per_page])) if per_page else None,
        "cer_bag": float(np.mean([p["cer_bag"] for p in per_page])) if per_page else None,
        "digit_exact_pages": (float(np.mean([p["digit_exact"] for p in per_page]))
                              if per_page else None),
        "invented": sum(p["invented"] for p in per_page),
        "invented_digits": sum(p["invented_digits"] for p in per_page),
        "seconds_per_page": (float(np.mean([p["pipeline_seconds"]
                                            for p in per_page])) if per_page else None),
        "queue_seconds_per_page": (float(np.mean([p["flags_seconds"]
                                                  for p in per_page])) if per_page else None),
        "digit_errors": errs_total,
        "engagement": {
            "candidates": sum(p["split"].get("candidates", 0) for p in per_page),
            "split": sum(p["split"].get("split", 0) for p in per_page),
            "segments": sum(p["split"].get("segments", 0) for p in per_page),
            "aborted": sum(p["split"].get("aborted", 0) for p in per_page),
        },
    }
    for k in (5, 10):
        hits = sum(p["queue"][f"hits@{k}"] for p in per_page)
        top = sum(p["queue"][f"top@{k}"] for p in per_page)
        summary[f"digit_recall@{k}"] = (round(hits / errs_total, 4)
                                        if errs_total else None)
        summary[f"digit_precision@{k}"] = round(hits / top, 4) if top else None
    return {"summary": summary, "pages": per_page}


def _gate(off: Dict, on: Dict, cer_target: Optional[float]) -> Dict:
    so, sn = off["summary"], on["summary"]
    d_cer = (sn["cer"] - so["cer"]) if so["cer"] is not None else None
    d_bag = (sn["cer_bag"] - so["cer_bag"]) if so["cer_bag"] is not None else None
    d_dex = (sn["digit_exact_pages"] - so["digit_exact_pages"])
    d_inv = sn["invented"] - so["invented"]
    d_inv_dig = sn["invented_digits"] - so["invented_digits"]
    d_sec = sn["seconds_per_page"] - so["seconds_per_page"]
    r10_off, r10_on = so.get("digit_recall@10"), sn.get("digit_recall@10")
    g = {
        "digit_r10_plus3pp": bool(r10_off is not None and r10_on is not None
                                  and r10_on >= r10_off + 0.03),
        "cer_not_worse": bool(d_cer is not None and d_cer <= 0.0),
        "bagcer_not_worse": bool(d_bag is not None and d_bag <= 0.0),
        "digit_exact_pages_not_worse": bool(d_dex >= 0.0),
        "invented_not_increased": bool(d_inv <= 0 and d_inv_dig <= 0),
        "latency_ok": bool(d_sec is not None and d_sec <= LATENCY_BUDGET_S),
    }
    if cer_target is not None:
        g["cer_at_or_below_target"] = bool(sn["cer"] is not None
                                           and sn["cer"] <= cer_target)
    delta = {"cer": d_cer, "cer_bag": d_bag, "digit_exact_pages": d_dex,
             "invented": d_inv, "invented_digits": d_inv_dig,
             "seconds_per_page": d_sec,
             "digit_errors": sn["digit_errors"] - so["digit_errors"],
             "digit_recall@10": (round(r10_on - r10_off, 4)
                                 if r10_off is not None and r10_on is not None
                                 else None)}
    g["pass"] = all(v for v in g.values() if isinstance(v, bool))
    return {"delta": delta, "clauses": g}


def run(data_dir: str, frozen: Optional[str], lang: str, limit: int,
        cer_target: Optional[float], identity: bool, json_path: Optional[str],
        repass_digits: bool = False) -> Dict:
    if frozen:
        sys.path.insert(0, os.path.join(BASE_DIR, "evals", "harness"))
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(frozen, data_dir=data_dir)
        if not ok:
            for p in problems[:5]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to evaluate")
        print(f"[split] frozen verified: {os.path.basename(frozen)}")

    rows = _load_rows(data_dir, limit=limit)
    print(f"[split] {len(rows)} pages from {os.path.basename(data_dir)}")

    arms: Dict[str, Dict] = {}
    for name, split in (("off", False), ("on", True)):
        arm = _run_arm(rows, lang, split, repass_digits=repass_digits)
        arms[name] = arm
        s = arm["summary"]
        print(f"  {name:<3}: CER {s['cer']:.4f} bagCER {s['cer_bag']:.4f} "
              f"digit-exact {s['digit_exact_pages']:.3f} "
              f"invented {s['invented']} "
              f"s/page {s['seconds_per_page']:.2f} "
              f"engagement {s['engagement']}", flush=True)

    changed_pages = sum(1 for a, b in zip(arms["off"]["pages"],
                                          arms["on"]["pages"])
                        if a["text"] != b["text"])
    result: Dict = {"set": os.path.basename(data_dir), "lang": lang,
                    "limit": limit, "identity": identity,
                    "changed_pages": changed_pages,
                    "arms": {k: v["summary"] for k, v in arms.items()}}
    if identity:
        result["clauses"] = {
            "zero_engagement_or_identical": bool(
                arms["on"]["summary"]["engagement"]["split"] == 0
                or changed_pages == 0),
        }
    else:
        result.update(_gate(arms["off"], arms["on"], cer_target))
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[split] wrote {json_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="merged number-run split gate")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cer-target", type=float, default=None,
                    help="e.g. 0.22 for the hard-10 clause")
    ap.add_argument("--identity", action="store_true",
                    help="non-target sets: only check zero engagement / "
                         "byte-identical text")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    run(args.data_dir, args.frozen, args.lang, args.limit, args.cer_target,
        args.identity, args.json)


if __name__ == "__main__":
    main()
