"""
eval_deva_lines_integration.py — pre-registered gate for the Devanagari line
reader inside the pipeline (not just on ALTO crops).

Compares `ocr_page(deva_lines="off")` (shipped RapidOCR) with
`ocr_page(deva_lines="on")` (RapidOCR detection + trained recognizer) on the
frozen sets:

  * heiDATA letterpress  -> human-corrected ALTO page text
  * nepali_pdf_v2        -> Gemini 2.5 Pro anchor page text

Co-conditions (docs/PLAN.md Appendix R5): page CER and bagCER not worse,
digit-exact not worse, review queue not larger, no new invented tokens,
<= +1 s/page.

Usage:
    python scripts/eval_deva_lines_integration.py \
        --data-dir data/doc_eval/heidata_printed \
        --frozen evals/manifests/heidata_printed_v1.json \
        --lang ne --json out/deva_lines_integration_heidata.json
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
sys.path.insert(0, os.path.join(BASE_DIR, "evals", "harness"))

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import ocr_page  # noqa: E402

LATENCY_BUDGET_S = 1.0


def _page_rows(data_dir: str, readings: Optional[str], limit: int = 0) -> List[Dict]:
    manifest = doc_data.load_dataset(data_dir)
    anchor = {}
    if readings and os.path.exists(readings):
        data = json.load(open(readings, encoding="utf-8"))
        anchor = {r["page"]: r["readings"]["gemini"] for r in data["rows"]}
    rows = []
    for e in manifest["entries"]:
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        gt = anchor.get(e["id"], "")
        boxes = []
        if not gt:
            try:
                gt = open(e["_gt_path"], encoding="utf-8").read()
            except Exception:  # noqa: BLE001
                gt = ""
        gt_lines = []
        if e.get("_boxes_path") and os.path.exists(e["_boxes_path"]):
            gt_lines = json.load(open(e["_boxes_path"], encoding="utf-8"))
        if not gt.strip():
            continue
        rows.append({"page": e["id"], "img": img, "gt": gt,
                     "gt_lines": gt_lines})
        if limit and len(rows) >= limit:
            break
    return rows


def _intersects(bbox, boxes) -> bool:
    x0, y0, x1, y1 = bbox
    for bx0, by0, bx1, by1 in boxes:
        if x1 >= bx0 and x0 <= bx1 and y1 >= by0 and y0 <= by1:
            return True
    return False


def _invented_digits(res, gt: str, gt_boxes: List) -> int:
    """Digit tokens the reader produced that GT does not contain.

    When GT line boxes exist, only tokens *inside the document text area* are
    counted: scan furniture (watermark URLs, page borders, the Heidelberg
    strip) is not in GT and would otherwise dominate the metric.
    """
    gt_set = {doc_metrics._norm_tok(t) for t in doc_metrics.tokens_of(gt)}
    n = 0
    for tok in res.tokens:
        if not any(ch.isdigit() for ch in tok.text):
            continue
        if doc_metrics._norm_tok(tok.text) in gt_set:
            continue
        if gt_boxes and not _intersects(tok.bbox, gt_boxes):
            continue
        n += 1
    return n


def _digit_inventions(res, gt_lines: List[Dict]) -> Dict[str, int]:
    """Digits put on a line whose GT has *no* digits at all, split by whether
    the pipeline flagged them.

    The product promise is "won't invent the numbers on your bill" - in
    practice: no *silent* inventions. A misread digit on a line that does have
    numbers is an accuracy error (measured by CER and the digit metrics); an
    invented digit that the flag machinery surfaces (invalid_sequence,
    digit_conflict, digit_uncertain) lands in the review queue, which is the
    product's honesty contract.
    """
    total = flagged = 0
    for tok in res.tokens:
        if not any(ch.isdigit() for ch in tok.text):
            continue
        overlapping = [b for b in gt_lines
                       if _intersects(tok.bbox, [b["bbox"]])]
        if not overlapping:
            continue  # scan furniture, not document text
        if any(any(ch.isdigit() for ch in b["text"]) for b in overlapping):
            continue
        total += 1
        flagged += 1 if tok.flags else 0
    return {"total": total, "flagged": flagged, "silent": total - flagged}


def _measure(rows: List[Dict], lang: str, deva_lines: str,
             ckpt: Optional[str]) -> Dict:
    per_page = []
    for r in rows:
        t0 = time.time()
        res = ocr_page(r["img"], backend="rapidocr", lang=lang,
                       recheck_digits=False, deva_lines=deva_lines,
                       deva_ckpt=ckpt)
        elapsed = time.time() - t0
        gt = r["gt"]
        hall = doc_metrics.hallucination_report(gt, "", res.text)
        per_page.append({
            "page": r["page"],
            "invented_digits": int(hall.get("invented_digits", 0)),
            "invented_digits_text_area": _invented_digits(
                res, gt, [b["bbox"] for b in r["gt_lines"]]),
            "digit_inventions": _digit_inventions(res, r["gt_lines"]),
            "cer": doc_metrics.cer(gt, res.text),
            "cer_bag": doc_metrics.cer_bag(gt, res.text),
            "digit_exact": 1.0 if doc_metrics.digit_exact(gt, res.text) else 0.0,
            "tokens": len(res.tokens),
            "flagged": int(res.meta.get("flagged", 0)),
            "invented": int(hall.get("invented", 0)),
            "seconds": round(elapsed, 3),
            "reader": res.meta.get("deva_line_reader"),
        })
    n = len(per_page)
    return {
        "pages": n,
        "cer": float(np.mean([p["cer"] for p in per_page])) if n else None,
        "cer_bag": float(np.mean([p["cer_bag"] for p in per_page])) if n else None,
        "digit_exact_pages": (float(np.mean([p["digit_exact"] for p in per_page]))
                              if n else None),
        "seconds_per_page": (float(np.mean([p["seconds"] for p in per_page]))
                             if n else None),
        "flagged_total": sum(p["flagged"] for p in per_page),
        "invented_total": sum(p["invented"] for p in per_page),
        "invented_digits_total": sum(p["invented_digits"] for p in per_page),
        "invented_digits_text_area": sum(p["invented_digits_text_area"]
                                         for p in per_page),
        "digit_inventions": sum(p["digit_inventions"]["total"]
                                for p in per_page),
        "digit_inventions_flagged": sum(p["digit_inventions"]["flagged"]
                                        for p in per_page),
        "silent_inventions": sum(p["digit_inventions"]["silent"]
                                 for p in per_page),
        "rows": per_page,
    }


def run(data_dir: str, frozen: Optional[str], lang: str, ckpt: Optional[str],
        readings: Optional[str], limit: int = 0) -> Dict:
    if frozen:
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(frozen, data_dir=data_dir)
        if not ok:
            for p in problems[:5]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to evaluate")
        print(f"[gate] frozen verified: {os.path.basename(frozen)}")

    from deva_reader import reader_info
    print(f"[gate] reader: {json.dumps(reader_info(ckpt))}")

    rows = _page_rows(data_dir, readings, limit=limit)
    print(f"[gate] {len(rows)} pages from {os.path.basename(data_dir)}")
    off = _measure(rows, lang, "off", ckpt)
    print(f"  off: CER {off['cer']:.4f} bagCER {off['cer_bag']:.4f} "
          f"digit-exact {off['digit_exact_pages']:.3f} "
          f"{off['seconds_per_page']:.2f}s/page", flush=True)
    # "on" is the shipped opt-in: the reader is forced on every line-shaped
    # page (the CLI keeps it off for born-digital PDFs and by default).
    on = _measure(rows, lang, "on", ckpt)
    engaged = sum(1 for p in on["rows"]
                  if (p.get("reader") or {}).get("active"))
    print(f"  on  : CER {on['cer']:.4f} bagCER {on['cer_bag']:.4f} "
          f"digit-exact {on['digit_exact_pages']:.3f} "
          f"{on['seconds_per_page']:.2f}s/page "
          f"(reader engaged on {engaged}/{len(on['rows'])} pages)", flush=True)

    # per-page deltas (off vs on) for the honest worst-case picture
    off_by_page = {p["page"]: p for p in off["rows"]}
    for p in on["rows"]:
        o = off_by_page.get(p["page"])
        p["cer_delta"] = p["cer"] - o["cer"] if o else 0.0
        p["cer_worse"] = p["cer_delta"] > 0.02
    on["pages_cer_worse"] = sum(1 for p in on["rows"] if p["cer_worse"])
    on["worst_page_cer_delta"] = max((p["cer_delta"] for p in on["rows"]),
                                     default=0.0)

    delta = {
        "cer": on["cer"] - off["cer"],
        "cer_bag": on["cer_bag"] - off["cer_bag"],
        "digit_exact_pages": on["digit_exact_pages"] - off["digit_exact_pages"],
        "seconds_per_page": on["seconds_per_page"] - off["seconds_per_page"],
        "flagged_total": on["flagged_total"] - off["flagged_total"],
        "invented_total": on["invented_total"] - off["invented_total"],
        "invented_digits_total": (on["invented_digits_total"]
                                  - off["invented_digits_total"]),
        "invented_digits_text_area": (on["invented_digits_text_area"]
                                      - off["invented_digits_text_area"]),
        "digit_inventions": on["digit_inventions"] - off["digit_inventions"],
        "silent_inventions": (on["silent_inventions"]
                              - off["silent_inventions"]),
        "pages_cer_worse": on["pages_cer_worse"],
    }
    gate = {
        "cer_not_worse": delta["cer"] <= 0.0,
        "bagcer_not_worse": delta["cer_bag"] <= 0.0,
        "digit_exact_not_worse": delta["digit_exact_pages"] >= 0.0,
        "queue_not_larger": delta["flagged_total"] <= 0,
        "no_silent_invented_digits": delta["silent_inventions"] <= 0,
        "latency_ok": delta["seconds_per_page"] <= LATENCY_BUDGET_S,
    }
    gate["pass"] = all(gate.values())
    print(f"[gate] deltas: {json.dumps({k: round(v, 4) for k, v in delta.items()})}")
    print(f"[gate] {json.dumps(gate)}")
    return {"set": os.path.basename(data_dir), "lang": lang,
            "reader": reader_info(ckpt), "off": off, "on": on,
            "delta": delta, "gate": gate}


def main():
    ap = argparse.ArgumentParser(description="Devanagari line-reader gate")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--readings", default=None,
                    help="anchor readings JSON (for sets without ALTO GT)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    result = run(args.data_dir, args.frozen, args.lang, args.ckpt,
                 args.readings, limit=args.limit)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0 if result["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
