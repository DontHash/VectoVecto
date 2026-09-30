"""
profile_textbook_scans.py — unlabeled behaviour profile for real textbook scans (W-D).

Cornell's Nepali textbooks have no text layer (DSpace TEXT derivatives are
112-276 bytes), so no CER can be computed for them. This harness reports what
is honest on real print: token density, flagged share, review-queue length,
script_mismatch rate, unknown_word rate (when the lexicon is installed),
digit-token share and seconds/page. No pipeline artifacts are written and the
text is never modified.

Usage:
    python scripts/profile_textbook_scans.py \
        --data-dir data/doc_eval/nepali_textbook_scans \
        --max-pages-per-pdf 5 --json out/textbook_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from document_ocr import review_queue  # noqa: E402
from document_pipeline import run_document_pipeline  # noqa: E402


def page_profile(result) -> Dict:
    """Flag/queue/digit stats for one pipeline result (no text claims)."""
    tokens = result.ocr.tokens
    n = len(tokens)
    return {
        "tokens": n,
        "flagged": sum(1 for t in tokens if t.flags),
        "queue": len(review_queue(tokens)),
        "script_mismatch": sum(1 for t in tokens if "script_mismatch" in t.flags),
        "unknown_word": sum(1 for t in tokens if "unknown_word" in t.flags),
        "digit_tokens": sum(1 for t in tokens if t.has_digits),
        "orientation_suspect": bool(result.meta.get("orientation_suspect")),
        "seconds": result.meta["seconds"],
        "lexicon": result.ocr.meta.get("lexicon"),
    }


def _share(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def aggregate(rows: List[Dict]) -> Dict:
    """Per-book and overall means over page rows (pure; unit-tested)."""
    if not rows:
        return {"pages": 0, "tokens": 0}
    tokens = sum(r["tokens"] for r in rows)
    return {
        "pages": len(rows),
        "tokens": tokens,
        "tokens_per_page": round(tokens / len(rows), 1),
        "flagged_share": _share(sum(r["flagged"] for r in rows), tokens),
        "queue_share": _share(sum(r["queue"] for r in rows), tokens),
        "script_mismatch_share": _share(
            sum(r["script_mismatch"] for r in rows), tokens),
        "unknown_word_share": _share(
            sum(r["unknown_word"] for r in rows), tokens),
        "digit_token_share": _share(
            sum(r["digit_tokens"] for r in rows), tokens),
        "orientation_suspect_pages": sum(
            1 for r in rows if r["orientation_suspect"]),
        "seconds_per_page_median": round(
            statistics.median(r["seconds"] for r in rows), 2),
        "lexicon": next((r["lexicon"] for r in rows if r.get("lexicon")), None),
    }


def profile_pdf(path: str, lang: str, max_pages: int, dpi: int = 200,
                start_page: int = 0) -> Dict:
    rows = []
    for idx, img, _gt in doc_data.pdf_to_pages(path, dpi=dpi):
        if idx < start_page:
            continue
        if max_pages and len(rows) >= max_pages:
            break
        result = run_document_pipeline(img, backend="rapidocr", lang=lang)
        rows.append({"page": idx, **page_profile(result)})
        print(f"    p{idx:03d}: {rows[-1]['tokens']} tokens "
              f"{rows[-1]['seconds']}s", flush=True)
    return {"file": os.path.basename(path), "pages": rows,
            "summary": aggregate(rows)}


def main():
    ap = argparse.ArgumentParser(description="Profile real textbook scans")
    ap.add_argument("--data-dir", default=os.path.join(
        BASE_DIR, "data", "doc_eval", "nepali_textbook_scans"))
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--max-pages-per-pdf", type=int, default=5)
    ap.add_argument("--start-page", type=int, default=0,
                    help="skip cover/title pages before profiling")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-lexicon", action="store_true",
                    help="disable the optional unknown_word lexicon")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.no_lexicon:
        os.environ["VERISCRIPT_LEXICON"] = os.path.join(
            BASE_DIR, "data", "lexicon", "__disabled__.txt")

    pdfs = sorted(f for f in os.listdir(args.data_dir)
                  if f.lower().endswith(".pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {args.data_dir}")
    report = {"lang": args.lang, "dpi": args.dpi,
              "max_pages_per_pdf": args.max_pages_per_pdf,
              "start_page": args.start_page,
              "created": time.strftime("%Y-%m-%d %H:%M:%S"), "books": []}
    all_rows: List[Dict] = []
    for name in pdfs:
        print(f"[probe] {name}")
        book = profile_pdf(os.path.join(args.data_dir, name), args.lang,
                           args.max_pages_per_pdf, args.dpi, args.start_page)
        report["books"].append(book)
        all_rows.extend(book["pages"])
    report["overall"] = aggregate(all_rows)

    print("\n=== TEXTBOOK SCAN PROBE (unlabeled, no CER) ===")
    for book in report["books"]:
        s = book["summary"]
        print(f"  {book['file']:<28} {s['pages']:>3}p "
              f"tok/p {s['tokens_per_page']:>6} "
              f"flag {s['flagged_share']:.2%} queue {s['queue_share']:.2%} "
              f"mismatch {s['script_mismatch_share']:.2%} "
              f"oov {s['unknown_word_share']:.2%} "
              f"med {s['seconds_per_page_median']}s")
    o = report["overall"]
    print(f"  OVERALL {o['pages']}p tok/p {o['tokens_per_page']} "
          f"flag {o['flagged_share']:.2%} queue {o['queue_share']:.2%} "
          f"mismatch {o['script_mismatch_share']:.2%} "
          f"oov {o['unknown_word_share']:.2%} "
          f"med {o['seconds_per_page_median']}s lexicon={o['lexicon']}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[probe] wrote {args.json}")


if __name__ == "__main__":
    main()
