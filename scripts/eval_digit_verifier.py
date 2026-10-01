#!/usr/bin/env python3
"""Digit-verifier gate (N2/N5): can a candidate reader catch engine digit errors?

For every digit token on the frozen modern pages, read the same crop with the
candidate CRNN and compare script-normalized digit sequences against the
anchor ground truth:

  conflict        engine and candidate disagree on the digit sequence
  engine right    engine sequence is in the GT digit set
  candidate right candidate sequence is in the GT digit set

The pre-registered decision rule (PLAN.md Appendix AC): candidate precision
on conflicts >= 0.6 -> adopt flag-only; >= 0.8 -> test a gated replacement;
< 0.4 -> stop.

This is the measurement screen; the product-level effect (queue recall@10,
latency) is measured by wiring the candidate as a `digit_verifier` and running
`evals/harness/eval_flags.py`.

Usage:
    python scripts/eval_digit_verifier.py \
        --data-dir data/doc_eval/nepali_pdf_v2 \
        --frozen evals/manifests/nepali_pdf_v2.json \
        --ckpt weights/deva_crnn_h48w512.pt \
        --min-digits 3 --json out/digit_verifier.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.core import data as doc_data  # noqa: E402
from veriscript.document.ocr import _crop_with_pad, ocr_page  # noqa: E402

LATIN = "0123456789"
DEVA = "०१२३४५६७८९"
DEVA_SET = set(DEVA)
_LATIN_TO_DEVA = str.maketrans(LATIN, DEVA)


def digits_of(text: str) -> str:
    """Digit sequence, script-normalized to Devanagari (mirrors ocr._digits_of)."""
    import re
    raw = "".join(re.findall(r"\d+", text or ""))
    return raw.translate(_LATIN_TO_DEVA)


def gt_digit_set(gt_text: str, min_digits: int) -> set:
    """All maximal digit runs (>= min_digits) in the GT, script-normalized."""
    import re
    runs = re.findall(r"[०-९]{%d,}" % min_digits, (gt_text or "").translate(_LATIN_TO_DEVA))
    return set(runs)


def _page_path(entry: Dict) -> Optional[str]:
    for key in ("_degraded_path", "_clean_path", "_image_path"):
        path = entry.get(key)
        if path and os.path.exists(path):
            return path
    return None


def _load_pages(data_dir: str, limit: int = 0) -> List[Dict]:
    manifest = doc_data.load_dataset(data_dir)
    rows = []
    for entry in manifest["entries"]:
        img_path = _page_path(entry)
        gt_path = entry.get("_gt_path")
        if not img_path or not gt_path or not os.path.exists(gt_path):
            continue
        img = doc_data.imread_safe(img_path)
        if img is None:
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = f.read()
        rows.append({"page": entry["id"], "img": img, "gt": gt})
        if limit and len(rows) >= limit:
            break
    return rows


def measure(rows: List[Dict], ckpt: Optional[str], min_digits: int,
            pad_ratio: float, lang: str) -> Dict:
    from veriscript.deva.reader import get_reader, resolve_ckpt
    reader = get_reader(ckpt)
    if reader is None:
        raise SystemExit(f"no checkpoint (tried {ckpt!r}); nothing to evaluate")

    per_page, details = [], []
    read_seconds = 0.0
    n_tokens = 0
    for row in rows:
        res = ocr_page(row["img"], backend="rapidocr", lang=lang,
                       recheck_digits=False, deva_lines="off")
        gt_set = gt_digit_set(row["gt"], min_digits)
        suspects = [t for t in res.tokens
                    if len(digits_of(t.text)) >= min_digits]
        crops = [_crop_with_pad(row["img"], t.bbox, pad_ratio)
                 for t in suspects]
        texts: List[str] = []
        if crops:
            t0 = time.time()
            texts = reader.read(crops)
            read_seconds += time.time() - t0
        n_tokens += len(suspects)
        page = {"page": row["page"], "tokens": len(suspects),
                "conflicts": 0, "engine_right": 0, "candidate_right": 0,
                "fixes": 0, "both_wrong": 0}
        for tok, text in zip(suspects, texts):
            seq, read_seq = digits_of(tok.text), digits_of(str(text))
            if not read_seq:  # mirror production: empty read = no conflict
                continue
            entry = {"page": row["page"], "engine": seq, "candidate": read_seq,
                     "engine_ok": seq in gt_set, "candidate_ok": read_seq in gt_set,
                     "flagged": bool(tok.flags)}
            if seq != read_seq:
                page["conflicts"] += 1
                if entry["engine_ok"] and not entry["candidate_ok"]:
                    page["engine_right"] += 1
                elif entry["candidate_ok"] and not entry["engine_ok"]:
                    page["candidate_right"] += 1
                    page["fixes"] += 1
                elif entry["candidate_ok"] and entry["engine_ok"]:
                    pass  # both readings valid (e.g. repeated numbers)
                else:
                    page["both_wrong"] += 1
                details.append(entry)
        per_page.append(page)

    conflicts = sum(p["conflicts"] for p in per_page)
    fixes = sum(p["fixes"] for p in per_page)
    engine_right = sum(p["engine_right"] for p in per_page)
    both_wrong = sum(p["both_wrong"] for p in per_page)
    precision = (fixes / conflicts) if conflicts else None
    return {
        "pages": len(per_page),
        "tokens": n_tokens,
        "conflicts": conflicts,
        "engine_right_on_conflicts": engine_right,
        "candidate_right_on_conflicts": fixes,
        "both_wrong_on_conflicts": both_wrong,
        "candidate_precision_on_conflicts": precision,
        "reader_seconds": round(read_seconds, 2),
        "seconds_per_token": round(read_seconds / n_tokens, 4) if n_tokens else None,
        "reader": {"active": True, "ckpt": os.path.basename(resolve_ckpt(ckpt) or "")},
        "per_page": per_page,
        "conflict_details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Digit-verifier screen (N2/N5)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--min-digits", type=int, default=3)
    ap.add_argument("--pad-ratio", type=float, default=0.08,
                    help="production verifier pad ratio (ocr.apply_digit_verifier)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.frozen:
        sys.path.insert(0, os.path.join(BASE_DIR, "evals", "harness"))
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(args.frozen, data_dir=args.data_dir)
        if not ok:
            for p in problems[:5]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to evaluate")
        print(f"[verifier] frozen verified: {os.path.basename(args.frozen)}")

    rows = _load_pages(args.data_dir, limit=args.limit)
    print(f"[verifier] {len(rows)} pages from {os.path.basename(args.data_dir)}")
    result = measure(rows, args.ckpt, args.min_digits, args.pad_ratio, args.lang)
    print(f"[verifier] tokens>={args.min_digits} digits: {result['tokens']} | "
          f"conflicts: {result['conflicts']}")
    print(f"[verifier] conflicts: engine right "
          f"{result['engine_right_on_conflicts']}, candidate fixes "
          f"{result['candidate_right_on_conflicts']}, both wrong "
          f"{result['both_wrong_on_conflicts']}")
    print(f"[verifier] candidate precision on conflicts: "
          f"{result['candidate_precision_on_conflicts']} "
          f"(>=0.6 flag-only, >=0.8 replacement, <0.4 stop)")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
