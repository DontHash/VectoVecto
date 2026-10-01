#!/usr/bin/env python3
"""eval_date_flags.py — pre-registered gate for the impossible-date validator
(Appendix AH).

Compares `date_flags` off/on with the queue metric definitions of
`eval_flags` (`ocr_page` with the shipped Devanagari repass, then
`review_queue`), and reports the validator's flag precision (flagged digit
tokens that are true digit errors per GT), text identity (flag-only proof) and
latency.

Usage:
    python scripts/eval_date_flags.py \
        --data-dir data/doc_eval/nepali_pdf_v2 \
        --frozen evals/manifests/nepali_pdf_v2.json --role target \
        --json out/date_flags_v2.json
    python scripts/eval_date_flags.py \
        --data-dir data/doc_eval/heidata_printed \
        --frozen evals/manifests/heidata_printed_v1.json --role other \
        --json out/date_flags_heidata.json
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

from veriscript.core import data as doc_data  # noqa: E402
from veriscript.core import metrics as doc_metrics  # noqa: E402
from veriscript.document.ocr import ocr_page, review_queue  # noqa: E402
from eval_number_split import _digit_tokens, _queue_metrics  # noqa: E402

LATENCY_BUDGET_S = 0.1


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


def _run_arm(rows: List[Dict], lang: str, date_flags: bool) -> Dict:
    per_page = []
    for r in rows:
        t0 = time.time()
        res = ocr_page(r["img"], backend="rapidocr", lang=lang,
                       recheck_digits=True, date_flags=date_flags)
        seconds = time.time() - t0
        q = _queue_metrics(res.tokens, r["gt"])
        gt_digits = {doc_metrics.digit_string(t) for t in _digit_tokens(r["gt"])}
        gt_digits.discard("")
        flagged_digits, true_flagged = 0, 0
        for t in res.tokens:
            if "invalid_format" not in t.flags:
                continue
            if t.has_digits:
                flagged_digits += 1
                d = doc_metrics.digit_string(t.text)
                if d and d not in gt_digits:
                    true_flagged += 1
        per_page.append({"page": r["page"], "queue": q, "text": res.text,
                         "queue_len": len(review_queue(res.tokens)),
                         "flagged": sum(1 for t in res.tokens
                                        if "invalid_format" in t.flags),
                         "flagged_digits": flagged_digits,
                         "true_flagged": true_flagged,
                         "seconds": seconds})
    errs_total = sum(p["queue"]["errors"] for p in per_page)
    summary = {
        "pages": len(per_page),
        "digit_errors": errs_total,
        "queue_len": sum(p["queue_len"] for p in per_page),
        "flagged": sum(p["flagged"] for p in per_page),
        "flagged_digits": sum(p["flagged_digits"] for p in per_page),
        "true_flagged": sum(p["true_flagged"] for p in per_page),
        "seconds_per_page": (float(np.mean([p["seconds"] for p in per_page]))
                             if per_page else None),
    }
    summary["flag_precision"] = (round(summary["true_flagged"]
                                       / summary["flagged_digits"], 4)
                                 if summary["flagged_digits"] else None)
    summary["flag_digit_recall"] = (round(summary["true_flagged"] / errs_total, 4)
                                    if errs_total else None)
    for k in (5, 10):
        hits = sum(p["queue"][f"hits@{k}"] for p in per_page)
        top = sum(p["queue"][f"top@{k}"] for p in per_page)
        summary[f"digit_recall@{k}"] = (round(hits / errs_total, 4)
                                        if errs_total else None)
        summary[f"digit_precision@{k}"] = round(hits / top, 4) if top else None
    return {"summary": summary, "pages": per_page}


def _gate(off: Dict, on: Dict, role: str) -> Dict:
    so, sn = off["summary"], on["summary"]
    r10o, r10n = so.get("digit_recall@10"), sn.get("digit_recall@10")
    p10o, p10n = so.get("digit_precision@10"), sn.get("digit_precision@10")
    changed = sum(1 for a, b in zip(off["pages"], on["pages"])
                  if a["text"] != b["text"])
    d_sec = (sn["seconds_per_page"] - so["seconds_per_page"])
    new_flags = sn["flagged_digits"] - so["flagged_digits"]
    clauses = {
        "text_identical": bool(changed == 0),
        "latency_ok": bool(d_sec <= LATENCY_BUDGET_S),
    }
    if role == "target":
        clauses["digit_r10_plus3pp"] = bool(r10o is not None
                                            and r10n is not None
                                            and r10n >= r10o + 0.03)
        clauses["flag_precision_ge_0.5"] = bool(
            sn["flag_precision"] is not None and sn["flag_precision"] >= 0.5)
    else:
        clauses["digit_r10_not_worse"] = bool(
            r10o is None or (r10n is not None and r10n >= r10o))
        clauses["digit_p10_not_worse"] = bool(
            p10o is None or (p10n is not None and p10n >= p10o))
    clauses["pass"] = all(v for v in clauses.values() if isinstance(v, bool))
    return {"changed_pages": changed, "new_flagged_digits": new_flags,
            "delta_seconds_per_page": d_sec,
            "delta_digit_recall@10": (round(r10n - r10o, 4)
                                      if r10o is not None and r10n is not None
                                      else None),
            "clauses": clauses}


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
        print(f"[date] frozen verified: {os.path.basename(frozen)}")

    rows = _load_rows(data_dir, limit=limit)
    print(f"[date] {len(rows)} pages from {os.path.basename(data_dir)}")
    arms = {}
    for name, enabled in (("off", False), ("on", True)):
        arm = _run_arm(rows, lang, enabled)
        arms[name] = arm
        s = arm["summary"]
        print(f"  {name:<3}: digit R@10 {s['digit_recall@10']} "
              f"P@10 {s['digit_precision@10']} errors {s['digit_errors']} "
              f"queue {s['queue_len']} flagged {s['flagged_digits']} "
              f"(true {s['true_flagged']}, precision {s['flag_precision']}) "
              f"{s['seconds_per_page']:.2f}s/page", flush=True)

    result = {"set": os.path.basename(data_dir), "lang": lang, "limit": limit,
              "role": role, "arms": {k: v["summary"] for k, v in arms.items()}}
    result.update(_gate(arms["off"], arms["on"], role))
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[date] wrote {json_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="impossible-date flag gate")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--role", choices=("target", "other"), default="target")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run(args.data_dir, args.frozen, args.lang, args.limit, args.role, args.json)


if __name__ == "__main__":
    main()
