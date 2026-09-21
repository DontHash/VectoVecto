"""
eval_flags.py — Devanagari flag-signal quality on frozen sets (F3).

Answers the product question the bake-off left open: given that the engine
cannot read Devanagari digits, do the *flags* at least tell the user where to
look? Measures, per frozen set:

  * digit-token error recall@K / precision@K of the review queue (micro over
    pages with >=1 digit error; error = hypothesis digit token whose digit
    string is absent from the page's GT digit multiset - the P4b definition)
  * script_mismatch stats (Latin tokens on a Devanagari page) incl. precision
  * ECE of raw vs isotonic-calibrated confidence

Usage:
    python eval_flags.py --data-dir data/doc_eval/heidata_printed --lang ne \
        --frozen evals/manifests/heidata_printed_v1.json \
        --json out/flags_heidata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import apply_digit_verifier, ocr_page, review_queue  # noqa: E402

DEVA_DIGITS = {chr(0x0966 + i) for i in range(10)}


def _digit_tokens(text: str) -> List[str]:
    return [t for t in doc_metrics.tokens_of(text)
            if any(c in DEVA_DIGITS or c.isdigit() for c in t)]


def run(data_dir: str, lang: str, limit: int = 0,
        digit_verifier=None) -> Dict:
    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"]
    if limit:
        entries = entries[:limit]

    topk = (5, 10)
    r_hits = {k: 0 for k in topk}
    r_total = 0
    p_hits = {k: 0 for k in topk}
    p_total = {k: 0 for k in topk}
    pages_with_errors = 0
    tokens_total = 0
    sm_total = sm_err = 0
    digit_tok_total = digit_tok_err = 0
    ece_raw, ece_cal = [], []
    rows = []
    for e in entries:
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        gt = open(e["_gt_path"], encoding="utf-8").read()
        gt_digits = {doc_metrics.digit_string(t) for t in _digit_tokens(gt)}
        gt_digits.discard("")
        rec = ocr_page(img, backend="rapidocr", lang=lang)
        vconflicts = 0
        if digit_verifier is not None:
            vconflicts = apply_digit_verifier(rec.tokens, img, digit_verifier)
        toks = rec.tokens
        tokens_total += len(toks)
        ece_raw.append(doc_metrics.ece(toks, gt))
        ece_cal.append(doc_metrics.ece(toks, gt, conf_attr="cal_conf"))

        errs = []
        for t in toks:
            if not t.has_digits:
                continue
            digit_tok_total += 1
            d = doc_metrics.digit_string(t.text)
            is_err = bool(d) and d not in gt_digits
            digit_tok_err += int(is_err)
            if is_err:
                errs.append(t)
            if "script_mismatch" in t.flags:
                sm_total += 1
                sm_err += int(doc_metrics._norm_tok(t.text) not in
                              {doc_metrics._norm_tok(x)
                               for x in doc_metrics.tokens_of(gt)})
        queue_len = 0
        if errs:
            pages_with_errors += 1
            r_total += len(errs)
            queue = review_queue(toks)
            queue_len = len(queue)
            err_ids = {id(t) for t in errs}
            for k in topk:
                top = queue[:k]
                p_hits[k] += sum(1 for t in top if id(t) in err_ids)
                p_total[k] += len(top)
                r_hits[k] += sum(1 for t in errs if t in top)
        rows.append({"page": e["id"], "digit_errors": len(errs),
                     "queue": queue_len, "verifier_conflicts": vconflicts,
                     "script_mismatch": sum(1 for t in toks
                                            if "script_mismatch" in t.flags)})
        print(f"  [{len(rows)}/{len(entries)}] {e['id']} "
              f"digit_errs={len(errs)} queue={queue_len}", flush=True)

    def _safe(num, den):
        return round(num / den, 4) if den else None

    summary = {
        "pages": len(rows),
        "tokens": tokens_total,
        "digit_tokens": digit_tok_total,
        "digit_token_error_rate": _safe(digit_tok_err, digit_tok_total),
        "pages_with_digit_errors": pages_with_errors,
        "verifier_conflicts": sum(r["verifier_conflicts"] for r in rows),
        **{f"digit_recall@{k}": _safe(r_hits[k], r_total) for k in topk},
        **{f"digit_precision@{k}": _safe(p_hits[k], p_total[k]) for k in topk},
        "script_mismatch_tokens": sm_total,
        "script_mismatch_rate": _safe(sm_total, tokens_total),
        "script_mismatch_precision": _safe(sm_err, sm_total),
        "ece_raw": round(float(np.mean(ece_raw)), 4) if ece_raw else None,
        "ece_isotonic": round(float(np.mean(ece_cal)), 4) if ece_cal else None,
    }
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Flag quality on frozen sets")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--digit-verifier", choices=["off", "bodhan"], default="off",
                    dest="digit_verifier",
                    help="optional second-model digit check before scoring")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.frozen:
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(args.frozen, data_dir=args.data_dir)
        if not ok:
            for p in problems[:10]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to score")
        print(f"[flags] frozen verified: {os.path.basename(args.frozen)}")

    verifier = None
    if args.digit_verifier != "off":
        from document_verifier import get_digit_verifier
        verifier = get_digit_verifier(args.digit_verifier)
        print(f"[flags] digit verifier enabled: {args.digit_verifier}")

    result = run(args.data_dir, args.lang, args.limit, digit_verifier=verifier)
    if verifier is not None:
        verifier.close()
    s = result["summary"]
    print("\n=== FLAG QUALITY (frozen) ===")
    for k, v in s.items():
        print(f"  {k:<28} {v}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"data_dir": args.data_dir, "lang": args.lang, **result},
                      f, indent=2)
        print(f"[flags] wrote {args.json}")


if __name__ == "__main__":
    main()
