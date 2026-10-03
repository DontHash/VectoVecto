#!/usr/bin/env python3
"""eval_multi_read.py — pre-registered gate for the multi-read panel
(track A of docs/HARNESS_PLAN.md §3).

Measures `multi_read` off/on with the exact definitions the gate names:

  * pipeline page CER / bagCER / digit-exact / invented: `run_document_pipeline`
    + `doc_metrics` (re-pass per the shipped Devanagari policy)
  * digit-token queue recall@K / precision@K: the AG/AH gate convention
    (`eval_number_split._queue_metrics`: `ocr_page` with the shipped
    Devanagari re-pass, `review_queue` order, top-K over all pages)
  * `multi_read_conflict` flag precision against the page GT digit set
  * output identity (pages whose OCR text changed) and latency

Roles:

  target  v2 digit queue R@10 >= baseline + 3 pp; flag precision >= 0.5;
          CER / bagCER / digit-exact not worse; invented not increased;
          text identical; <= +1 s/page
  other   non-target sets: digit R@10 / P@10 not worse; queue not larger;
          text identical; <= +1 s/page

Usage:
    python scripts/eval_multi_read.py \
        --data-dir data/doc_eval/nepali_pdf_v2 \
        --frozen evals/manifests/nepali_pdf_v2.json --limit 10 --role target \
        --json out/multi_read_hard10.json
    python scripts/eval_multi_read.py \
        --data-dir data/doc_eval/heidata_printed \
        --frozen evals/manifests/heidata_printed_v1.json --role other
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from veriscript.core import metrics as doc_metrics  # noqa: E402
from veriscript.document.ocr import ocr_page  # noqa: E402
from eval_number_split import (_digit_tokens, _load_rows,  # noqa: E402
                               _queue_metrics)

LATENCY_BUDGET_S = 1.0


def _run_arm(rows: List[Dict], lang: str, multi_read: str) -> Dict:
    from veriscript.document.pipeline import run_document_pipeline

    per_page = []
    for r in rows:
        t0 = time.time()
        pres = run_document_pipeline(r["img"], backend="rapidocr", lang=lang,
                                     repass_digits=None, multi_read=multi_read)
        pipeline_s = time.time() - t0
        gt = r["gt"]
        hall = doc_metrics.hallucination_report(gt, "", pres.ocr.text)
        t1 = time.time()
        res = ocr_page(r["img"], backend="rapidocr", lang=lang,
                       recheck_digits=True, multi_read=multi_read)
        flags_s = time.time() - t1
        q = _queue_metrics(res.tokens, gt)
        gt_digits = {doc_metrics.digit_string(t) for t in _digit_tokens(gt)}
        gt_digits.discard("")
        mr_flagged = mr_true = 0
        for t in res.tokens:
            if "multi_read_conflict" not in t.flags:
                continue
            mr_flagged += 1
            d = doc_metrics.digit_string(t.text)
            if d and d not in gt_digits:
                mr_true += 1
        per_page.append({
            "page": r["page"],
            "cer": doc_metrics.cer(gt, pres.ocr.text),
            "cer_bag": doc_metrics.cer_bag(gt, pres.ocr.text),
            "digit_exact": (1.0 if doc_metrics.digit_exact(gt, pres.ocr.text)
                            else 0.0),
            "invented": int(hall.get("invented", 0)),
            "invented_digits": int(hall.get("invented_digits", 0)),
            "pipeline_seconds": round(pipeline_s, 3),
            "flags_seconds": round(flags_s, 3),
            "queue": q,
            "multi_read": res.meta.get("multi_read") or {},
            "mr_flagged": mr_flagged,
            "mr_true": mr_true,
            "text": res.text,
        })

    errs_total = sum(p["queue"]["errors"] for p in per_page)
    queue_total = sum(p["queue"]["queue"] for p in per_page)
    mr_flagged = sum(p["mr_flagged"] for p in per_page)
    mr_true = sum(p["mr_true"] for p in per_page)
    engagement = {k: sum(p["multi_read"].get(k, 0) for p in per_page)
                  for k in ("checked", "agree", "split", "unreadable",
                            "conflicts")}
    summary = {
        "pages": len(per_page),
        "cer": float(np.mean([p["cer"] for p in per_page])) if per_page else None,
        "cer_bag": (float(np.mean([p["cer_bag"] for p in per_page]))
                    if per_page else None),
        "digit_exact_pages": (float(np.mean([p["digit_exact"]
                                            for p in per_page]))
                              if per_page else None),
        "invented": sum(p["invented"] for p in per_page),
        "invented_digits": sum(p["invented_digits"] for p in per_page),
        "seconds_per_page": (float(np.mean([p["pipeline_seconds"]
                                            for p in per_page]))
                             if per_page else None),
        "queue_seconds_per_page": (float(np.mean([p["flags_seconds"]
                                                  for p in per_page]))
                                   if per_page else None),
        "digit_errors": errs_total,
        "queue_len": queue_total,
        "mr_flagged": mr_flagged,
        "mr_true": mr_true,
        "mr_precision": (round(mr_true / mr_flagged, 4)
                         if mr_flagged else None),
        "engagement": engagement,
    }
    for k in (5, 10):
        hits = sum(p["queue"][f"hits@{k}"] for p in per_page)
        top = sum(p["queue"][f"top@{k}"] for p in per_page)
        summary[f"digit_recall@{k}"] = (round(hits / errs_total, 4)
                                        if errs_total else None)
        summary[f"digit_precision@{k}"] = round(hits / top, 4) if top else None
    return {"summary": summary, "pages": per_page}


def _gate(off: Dict, on: Dict, role: str, changed_pages: int) -> Dict:
    so, sn = off["summary"], on["summary"]
    r10o, r10n = so.get("digit_recall@10"), sn.get("digit_recall@10")
    p10o, p10n = so.get("digit_precision@10"), sn.get("digit_precision@10")
    d_cer = (sn["cer"] - so["cer"]) if so["cer"] is not None else None
    d_bag = (sn["cer_bag"] - so["cer_bag"]) if so["cer_bag"] is not None else None
    d_dex = sn["digit_exact_pages"] - so["digit_exact_pages"]
    d_inv = sn["invented"] - so["invented"]
    d_inv_dig = sn["invented_digits"] - so["invented_digits"]
    d_sec = sn["seconds_per_page"] - so["seconds_per_page"]
    clauses = {
        "text_identical": bool(changed_pages == 0),
        "latency_ok": bool(d_sec is not None and d_sec <= LATENCY_BUDGET_S),
    }
    if role == "target":
        clauses["digit_r10_plus3pp"] = bool(
            r10o is not None and r10n is not None and r10n >= r10o + 0.03)
        clauses["flag_precision_ge_0.5"] = bool(
            sn["mr_precision"] is not None and sn["mr_precision"] >= 0.5)
        clauses["cer_not_worse"] = bool(d_cer is not None and d_cer <= 0.0)
        clauses["bagcer_not_worse"] = bool(d_bag is not None and d_bag <= 0.0)
        clauses["digit_exact_pages_not_worse"] = bool(d_dex >= 0.0)
        clauses["invented_not_increased"] = bool(d_inv <= 0 and d_inv_dig <= 0)
    else:
        clauses["digit_r10_not_worse"] = bool(
            r10o is None or (r10n is not None and r10n >= r10o))
        clauses["digit_p10_not_worse"] = bool(
            p10o is None or (p10n is not None and p10n >= p10o))
        clauses["queue_not_larger"] = bool(sn["queue_len"] <= so["queue_len"])
    clauses["pass"] = all(v for v in clauses.values() if isinstance(v, bool))
    delta = {
        "cer": d_cer, "cer_bag": d_bag, "digit_exact_pages": d_dex,
        "invented": d_inv, "invented_digits": d_inv_dig,
        "seconds_per_page": d_sec,
        "digit_errors": sn["digit_errors"] - so["digit_errors"],
        "queue_len": sn["queue_len"] - so["queue_len"],
        "digit_recall@10": (round(r10n - r10o, 4)
                            if r10o is not None and r10n is not None else None),
        "digit_precision@10": (round(p10n - p10o, 4)
                               if p10o is not None and p10n is not None else None),
    }
    return {"changed_pages": changed_pages, "delta": delta, "clauses": clauses}


def run(data_dir: str, frozen: Optional[str], lang: str, limit: int,
        role: str, json_path: Optional[str]) -> Dict:
    if frozen:
        sys.path.insert(0, os.path.join(BASE_DIR, "evals", "harness"))
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(frozen, data_dir=data_dir)
        if not ok:
            for p in problems[:5]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to evaluate")
        print(f"[multi-read] frozen verified: {os.path.basename(frozen)}")

    rows = _load_rows(data_dir, limit=limit)
    print(f"[multi-read] {len(rows)} pages from {os.path.basename(data_dir)}")

    arms: Dict[str, Dict] = {}
    for name, mode in (("off", "off"), ("on", "panel")):
        arm = _run_arm(rows, lang, mode)
        arms[name] = arm
        s = arm["summary"]
        print(f"  {name:<3}: R@10 {s['digit_recall@10']} "
              f"P@10 {s['digit_precision@10']} "
              f"queue {s['queue_len']} flags {s['mr_flagged']} "
              f"(true {s['mr_true']}, precision {s['mr_precision']}) "
              f"engagement {s['engagement']} "
              f"s/page {s['seconds_per_page']:.2f}", flush=True)

    changed_pages = sum(1 for a, b in zip(arms["off"]["pages"],
                                          arms["on"]["pages"])
                        if a["text"] != b["text"])
    result: Dict = {"set": os.path.basename(data_dir), "lang": lang,
                    "limit": limit, "role": role,
                    "arms": {k: v["summary"] for k, v in arms.items()}}
    result.update(_gate(arms["off"], arms["on"], role, changed_pages))
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[multi-read] wrote {json_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="multi-read panel gate (track A)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--role", choices=("target", "other"), default="target")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    result = run(args.data_dir, args.frozen, args.lang, args.limit, args.role,
                 args.json)
    print(f"[multi-read] PASS={result['clauses']['pass']} "
          f"clauses={result['clauses']}")


if __name__ == "__main__":
    main()
