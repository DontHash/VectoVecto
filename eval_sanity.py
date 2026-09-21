"""
eval_sanity.py — behavior audit on unlabeled real pages (no ground truth).

For sets without usable transcriptions (archive.org Nepali scans), CER is not
measurable. What *is* measurable: does the shipped pipeline behave sanely —
orientation decisions, suspect flags, reading-order churn, review-queue rate,
latency. This is the honest use of unlabeled real data; no CER may be quoted.

Usage:
    python eval_sanity.py --data-dir data/doc_eval/nepali_unlabeled \
        --lang ne --json out/nepali_unlabeled.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402


def run_audit(entries: List[Dict], pipeline_fn, lang: str | None = None,
              limit: int = 0) -> Dict:
    """Run the shipped pipeline per page; collect behavior, never text scores."""
    rows = []
    for e in entries:
        if limit and len(rows) >= limit:
            break
        img = doc_data.imread_safe(e["_image_path"])
        if img is None:
            print(f"  [skip] missing {e['_image_path']}")
            continue
        try:
            res = pipeline_fn(img, backend="rapidocr", lang=lang,
                              make_pdf=False, make_overlay=False,
                              make_txt=False, make_json=False)
        except TypeError:
            # test fakes may accept fewer keywords
            res = pipeline_fn(img)
        meta = getattr(res, "meta", {}) or {}
        rows.append({
            "id": e["id"],
            "angle": meta.get("auto_rotate"),
            "evidence": meta.get("orientation_evidence"),
            "suspect": bool(meta.get("orientation_suspect")),
            "tokens": len(getattr(getattr(res, "ocr", None), "tokens", []) or []),
            "flagged": sum(1 for t in (getattr(getattr(res, "ocr", None), "tokens", []) or [])
                           if t.flags),
            "splits": meta.get("reading_order_splits"),
            "changed": bool(meta.get("reading_order_changed")),
            "seconds": meta.get("seconds"),
        })
        print(f"  [{len(rows)}] {e['id']} angle={rows[-1]['angle']} "
              f"suspect={rows[-1]['suspect']} tokens={rows[-1]['tokens']}",
              flush=True)

    def _frac(pred):
        return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else None

    angles = {}
    for r in rows:
        key = str(r["angle"])
        angles[key] = angles.get(key, 0) + 1
    summary = {
        "pages": len(rows),
        "angle_histogram": dict(sorted(angles.items())),
        "suspect_rate": _frac(lambda r: r["suspect"]),
        "identity_rate": _frac(lambda r: not r["changed"]),
        "flagged_rate": _frac(lambda r: r["flagged"] > 0),
        "tokens_median": (statistics.median([r["tokens"] for r in rows])
                          if rows else None),
        "seconds_median": (statistics.median([r["seconds"] for r in rows
                                              if r["seconds"] is not None])
                           if any(r["seconds"] is not None for r in rows) else None),
    }
    return {"summary": summary, "rows": rows}


def print_report(result: Dict) -> None:
    s = result["summary"]
    print("\n=== UNLABELED SANITY AUDIT (no CER - behavior only) ===")
    print(f"pages            : {s['pages']}")
    print(f"angle histogram  : {s['angle_histogram']}")
    if s["suspect_rate"] is not None:
        print(f"orientation suspect rate : {s['suspect_rate']:.3f}")
        print(f"reading-order identity   : {s['identity_rate']:.3f}")
        print(f"pages with flags         : {s['flagged_rate']:.3f}")
    print(f"tokens/page median       : {s['tokens_median']}")
    if s["seconds_median"] is not None:
        print(f"seconds/page median      : {s['seconds_median']:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Behavior audit on unlabeled pages")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest["entries"]
    for e in entries:
        e["_image_path"] = os.path.join(args.data_dir, e["image"])
    if args.limit:
        entries = entries[:args.limit]
    print(f"[sanity] {len(entries)} pages from {args.data_dir}")

    from document_pipeline import run_document_pipeline
    result = run_audit(entries, run_document_pipeline, lang=args.lang)
    print_report(result)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[sanity] wrote {args.json}")


if __name__ == "__main__":
    main()
