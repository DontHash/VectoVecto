"""
eval_error_taxonomy.py — where Devanagari recognition fails, per book (R1).

CER says how much; this says what. Token errors between the human-corrected
ALTO GT (heiDATA letterpress, trustworthy) and RapidOCR are classified as
digit / matra / consonant / order / segmentation / missing / invented /
other, aggregated per book. OCR tokens are assigned to ALTO lines by geometry,
so the classification is line-scoped; tokens matching no line are counted as
`detection_extra`.

Usage:
    python evals/harness/eval_error_taxonomy.py --data-dir data/doc_eval/heidata_printed \
        --frozen evals/manifests/heidata_printed_v1.json --lang ne \
        --json out/error_taxonomy_heidata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import ocr_page  # noqa: E402

ERROR_KINDS = ("digit", "matra", "consonant", "order", "segmentation",
               "missing", "invented", "other")

DEVA_MATRAS = frozenset(
    [chr(c) for c in range(0x093E, 0x094D)]
    + [chr(c) for c in range(0x094E, 0x0950)]
    + [chr(c) for c in range(0x0955, 0x0958)]
    + [chr(c) for c in range(0x0962, 0x0964)])
DEVA_CONSONANTS = frozenset(
    [chr(c) for c in range(0x0915, 0x093A)]
    + [chr(c) for c in range(0x0958, 0x0960)]
    + [chr(c) for c in range(0x0978, 0x0980)])


def _token_kind(gt_tok: str, hyp_tok: str) -> str:
    g, h = doc_metrics._norm_tok(gt_tok), doc_metrics._norm_tok(hyp_tok)
    gd, hd = doc_metrics.digit_string(g), doc_metrics.digit_string(h)
    if (gd or hd) and gd != hd:
        return "digit"
    if sorted(g) == sorted(h):
        return "order"
    if not (set(g) & set(h)):
        return "unrelated"  # counted as missing + invented
    g_m = {c for c in g if c in DEVA_MATRAS}
    h_m = {c for c in h if c in DEVA_MATRAS}
    if g_m != h_m:
        return "matra"
    g_c = {c for c in g if c in DEVA_CONSONANTS}
    h_c = {c for c in h if c in DEVA_CONSONANTS}
    if g_c != h_c:
        return "consonant"
    return "other"


def classify_token_errors(gt: str, hyp: str) -> Dict[str, int]:
    """Classify token-level differences between one GT line and one OCR line."""
    gt_toks = [t for t in (doc_metrics._norm_tok(x)
                           for x in doc_metrics.tokens_of(gt)) if t]
    hyp_toks = [t for t in (doc_metrics._norm_tok(x)
                            for x in doc_metrics.tokens_of(hyp)) if t]
    counts = {k: 0 for k in ERROR_KINDS}
    counts["gt_tokens"] = len(gt_toks)
    counts["hyp_tokens"] = len(hyp_toks)
    sm = SequenceMatcher(a=gt_toks, b=hyp_toks, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        g, h = gt_toks[i1:i2], hyp_toks[j1:j2]
        if op == "equal":
            continue
        if op == "delete":
            counts["missing"] += len(g)
        elif op == "insert":
            counts["invented"] += len(h)
        elif len(g) == 1 and len(h) == 1:
            kind = _token_kind(g[0], h[0])
            if kind == "unrelated":
                counts["missing"] += 1
                counts["invented"] += 1
            else:
                counts[kind] += 1
        else:
            counts["segmentation"] += 1
    return counts


def _assign_lines(boxes: List[Dict], tokens) -> Dict[int, List[str]]:
    """Map OCR tokens to GT line indices by token-center containment."""
    by_line: Dict[int, List[str]] = {}
    for t in tokens:
        x0, y0, x1, y1 = t.bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best = None
        for i, b in enumerate(boxes):
            bx0, by0, bx1, by1 = b["bbox"]
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                best = i
                break
        if best is None:
            best = -1
        by_line.setdefault(best, []).append(t.text)
    return by_line


def run(data_dir: str, lang: str = "ne", limit: int = 0,
        books: Optional[List[str]] = None) -> Dict:
    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"]
    if books:
        entries = [e for e in entries if e.get("book") in books]
    if limit:
        entries = entries[:limit]

    per_book: Dict[str, Dict] = {}
    rows = []
    for i, e in enumerate(entries, 1):
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        boxes = json.load(open(e["_boxes_path"], encoding="utf-8"))
        rec = ocr_page(img, backend="rapidocr", lang=lang)
        by_line = _assign_lines(boxes, rec.tokens)
        page_counts = {k: 0 for k in ERROR_KINDS}
        page_counts["detection_extra"] = 0
        for i_line, b in enumerate(boxes):
            hyp_text = " ".join(by_line.get(i_line, []))
            c = classify_token_errors(b["text"], hyp_text)
            for k in ERROR_KINDS:
                page_counts[k] += c[k]
            page_counts["gt_tokens"] = page_counts.get("gt_tokens", 0) + c["gt_tokens"]
        page_counts["detection_extra"] = len(by_line.get(-1, []))
        page_counts["lines"] = len(boxes)
        book = e.get("book", "?")
        agg = per_book.setdefault(book, {k: 0 for k in ERROR_KINDS})
        agg["lines"] = agg.get("lines", 0) + len(boxes)
        agg["gt_tokens"] = agg.get("gt_tokens", 0) + page_counts["gt_tokens"]
        agg["detection_extra"] = (agg.get("detection_extra", 0)
                                  + page_counts["detection_extra"])
        for k in ERROR_KINDS:
            agg[k] += page_counts[k]
        rows.append({"page": e["id"], "book": book, **page_counts})
        print(f"  [{i}/{len(entries)}] {e['id'][:36]:<38} "
              f"errors {sum(page_counts[k] for k in ERROR_KINDS)}", flush=True)

    for book, agg in per_book.items():
        denom = max(1, agg["gt_tokens"])
        agg["error_rate"] = round(sum(agg[k] for k in ERROR_KINDS) / denom, 4)
        agg["share"] = {k: (round(agg[k] / max(1, sum(agg[x]
                                                      for x in ERROR_KINDS)), 4))
                        for k in ERROR_KINDS}
    return {"summary": {"books": per_book, "pages": len(rows)}, "rows": rows}


def print_report(result: Dict) -> None:
    print("\n=== ERROR TAXONOMY (per book) ===")
    header = (f"{'book':<24}{'lines':>6}{'tok':>6}{'err':>6}{'rate':>7}"
              f"{'digit':>7}{'matra':>7}{'cons':>7}{'order':>7}{'seg':>6}"
              f"{'miss':>6}{'inv':>6}{'other':>7}{'det_x':>7}")
    print(header)
    print("-" * len(header))
    for book, a in sorted(result["summary"]["books"].items()):
        print(f"{book[:22]:<24}{a['lines']:>6}{a['gt_tokens']:>6}"
              f"{sum(a[k] for k in ERROR_KINDS):>6}{a['error_rate']:>7.3f}"
              f"{a['digit']:>7}{a['matra']:>7}{a['consonant']:>7}"
              f"{a['order']:>7}{a['segmentation']:>6}{a['missing']:>6}"
              f"{a['invented']:>6}{a['other']:>7}{a['detection_extra']:>7}")


def main():
    ap = argparse.ArgumentParser(description="Devanagari error taxonomy")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--books", nargs="*", default=None)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.frozen:
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(args.frozen, data_dir=args.data_dir)
        if not ok:
            for p in problems[:10]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to score")
        print(f"[taxonomy] frozen verified: {os.path.basename(args.frozen)}")

    result = run(args.data_dir, lang=args.lang, limit=args.limit,
                 books=args.books)
    print_report(result)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[taxonomy] wrote {args.json}")


if __name__ == "__main__":
    main()
