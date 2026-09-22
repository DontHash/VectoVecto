"""
eval_models.py — Devanagari OCR bake-off harness (frozen-set scoring).

Two modes:
  lines : recognition-only on line crops
          --lines-source heidata   (default: 150 digit-bearing ALTO lines cut
                                    from the frozen heiDATA pages, deterministic)
          --lines-source nepali_lines (the 500-crop slice; GT caveat applies)
  pages : whole-page text on a real manifest (heidata_printed / nepali_pdf)

All numbers are printed and written to JSON; 95% bootstrap CIs included.
The shipped path is untouched: candidates come from bakeoff_models.py.

Usage:
    python evals/harness/eval_models.py --model rapidocr --mode lines --lines-source heidata \
        --lang ne --json out/bakeoff_rapidocr_lines.json
    python evals/harness/eval_models.py --model glmocr --mode pages --data-dir \
        data/doc_eval/heidata_printed --lang ne --json out/bakeoff_glmocr_pages.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import bakeoff_models  # noqa: E402
import doc_data  # noqa: E402
import doc_metrics  # noqa: E402

HEIDATA_DEFAULT = os.path.join(BASE_DIR, "data", "doc_eval", "heidata_printed")
HEIDATA_LINES_DEFAULT = os.path.join(BASE_DIR, "data", "doc_eval",
                                     "heidata_digit_lines")


def iter_heidata_lines(data_dir: str, limit: int = 150,
                       pad: int = 4) -> List[Dict]:
    """Cut digit-bearing ALTO lines (first `limit`, manifest order, 4px pad)."""
    manifest = doc_data.load_dataset(data_dir)
    out: List[Dict] = []
    for e in manifest["entries"]:
        if limit and len(out) >= limit:
            break
        img = doc_data.imread_safe(e["_clean_path"])
        if img is None or not e.get("_boxes_path"):
            continue
        with open(e["_boxes_path"], encoding="utf-8") as f:
            boxes = json.load(f)
        h, w = img.shape[:2]
        for i, b in enumerate(boxes):
            if limit and len(out) >= limit:
                break
            gd = doc_metrics.digit_string(b["text"])
            if not gd:
                continue
            x0, y0, x1, y1 = b["bbox"]
            crop = img[max(0, y0 - pad):min(h, y1 + pad),
                       max(0, x0 - pad):min(w, x1 + pad)]
            if crop.size == 0:
                continue
            out.append({"id": f"{e['id']}_l{i:03d}", "crop": crop,
                        "gt": b["text"]})
    return out


def load_lines_source(source: str, data_dir: Optional[str],
                      limit: int = 0) -> List[Dict]:
    if source == "heidata":
        return iter_heidata_lines(data_dir or HEIDATA_DEFAULT,
                                  limit=limit or 150)
    if source == "nepali_lines":
        m = doc_data.load_line_dataset(data_dir or os.path.join(
            BASE_DIR, "data", "doc_eval", "nepali_lines"))
        entries = m["entries"]
        if limit:
            entries = entries[:limit]
        return [{"id": e["id"], "img_path": e["_image_path"],
                 "gt": open(e["_gt_path"], encoding="utf-8").read().strip()}
                for e in entries]
    raise SystemExit(f"unknown lines source {source!r}")


def score_lines(cand, ctx, items: List[Dict], lang: Optional[str]) -> Dict:
    crops = []
    for it in items:
        img = it.get("crop")
        if img is None:
            img = doc_data.imread_safe(it["img_path"])
        crops.append(img if img is not None else np.zeros((10, 10, 3), np.uint8))
    t0 = time.time()
    texts = cand.recognize_lines(ctx, crops, lang=lang)
    elapsed = time.time() - t0
    cers, exact, digit_total, digit_ok, empties = [], [], 0, 0, 0
    rows = []
    for it, hyp in zip(items, texts):
        gt = it["gt"]
        c = doc_metrics.cer(gt, hyp)
        cers.append(c)
        exact.append(1.0 if c == 0.0 else 0.0)
        if not (hyp or "").strip():
            empties += 1
        if doc_metrics.digit_string(gt):
            digit_total += 1
            ok = doc_metrics.digit_exact(gt, hyp)
            digit_ok += int(ok)
            rows.append({"id": it["id"], "cer": c, "gt": gt, "hyp": hyp,
                         "digit_exact": bool(ok)})
    summary = {
        "lines": len(items),
        "cer": float(np.mean(cers)) if cers else None,
        "cer_ci": [round(v, 4) for v in doc_metrics.bootstrap_ci(cers, seed=1)]
        if cers else None,
        "exact_match": float(np.mean(exact)) if exact else None,
        "digit_lines": digit_total,
        "digit_exact_rate": (digit_ok / digit_total) if digit_total else None,
        "empty_outputs": empties,
        "seconds_per_line": (elapsed / max(1, len(items))),
    }
    worst = sorted(rows, key=lambda r: -r["cer"])[:15]
    return {"summary": summary, "worst": worst}


def score_pages(cand, ctx, data_dir: str, lang: Optional[str],
                limit: int = 0, with_reference: bool = True) -> Dict:
    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"]
    if limit:
        entries = entries[:limit]
    ref = None
    if with_reference and cand.name != "rapidocr":
        from document_ocr import get_backend
        ref = get_backend("rapidocr")
    cers, bags, digit_cers, inventories, misses, secs = [], [], [], [], [], []
    invented_tot = invented_digits = 0
    rows = []
    for e in entries:
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        gt = open(e["_gt_path"], encoding="utf-8").read()
        t0 = time.time()
        hyp = cand.recognize_pages(ctx, [img], lang=lang)[0]
        secs.append(time.time() - t0)
        c = doc_metrics.cer(gt, hyp)
        b = doc_metrics.cer_bag(gt, hyp)
        cers.append(c)
        bags.append(b)
        dg = doc_metrics.digit_cer(gt, hyp, bag=True)
        digit_cers.append(dg)
        raw_text = hyp if ref is None else ref.run(img, lang=lang).text
        hall = doc_metrics.hallucination_report(gt, raw_text, hyp)
        invented_tot += hall.get("invented", 0)
        invented_digits += hall.get("invented_digits", 0)
        bs = doc_metrics.bag_stats(gt, hyp)
        inventories.append(bs["invented_rate"])
        misses.append(bs["miss_rate"])
        rows.append({"page": e["id"], "cer": c, "cer_bag": b,
                     "digit_cer_bag": dg, "invented_rate": bs["invented_rate"],
                     "invented_vs_gt_raw": hall.get("invented", 0),
                     "miss_rate": bs["miss_rate"], "seconds": secs[-1],
                     "hyp": hyp[:400]})
        print(f"  [{len(rows)}/{len(entries)}] {e['id']} CER {c:.3f}", flush=True)
    summary = {
        "pages": len(rows),
        "cer": float(np.mean(cers)) if cers else None,
        "cer_ci": [round(v, 4) for v in doc_metrics.bootstrap_ci(cers, seed=1)]
        if cers else None,
        "cer_bag": float(np.mean(bags)) if bags else None,
        "cer_bag_ci": [round(v, 4) for v in doc_metrics.bootstrap_ci(bags, seed=2)]
        if bags else None,
        "digit_cer_bag": float(np.mean(digit_cers)) if digit_cers else None,
        "invented_rate": float(np.mean(inventories)) if inventories else None,
        "invented_tokens": invented_tot,
        "invented_digits": invented_digits,
        "miss_rate": float(np.mean(misses)) if misses else None,
        "seconds_per_page": float(np.mean(secs)) if secs else None,
    }
    return {"summary": summary, "rows": rows}


def print_lines(summary: Dict) -> None:
    s = summary
    print("\n=== BAKE-OFF: LINES ===")
    print(f"lines              : {s['lines']}")
    if s["cer"] is not None:
        print(f"CER                : {s['cer']:.4f}"
              + (f"  CI [{s['cer_ci'][0]:.3f}-{s['cer_ci'][1]:.3f}]"
                 if s.get("cer_ci") else ""))
    print(f"exact match        : {s['exact_match']:.3f}")
    if s["digit_exact_rate"] is not None:
        print(f"digit-seq exact    : {s['digit_exact_rate']:.3f} "
              f"({s['digit_lines']} digit lines)")
    print(f"empty outputs      : {s['empty_outputs']}")
    print(f"seconds/line       : {s['seconds_per_line']:.3f}")


def print_pages(summary: Dict) -> None:
    s = summary
    print("\n=== BAKE-OFF: PAGES ===")
    print(f"pages        : {s['pages']}")
    print(f"CER          : {s['cer']:.4f}"
          + (f"  CI [{s['cer_ci'][0]:.3f}-{s['cer_ci'][1]:.3f}]"
             if s.get("cer_ci") else ""))
    print(f"bagCER       : {s['cer_bag']:.4f}"
          + (f"  CI [{s['cer_bag_ci'][0]:.3f}-{s['cer_bag_ci'][1]:.3f}]"
             if s.get("cer_bag_ci") else ""))
    print(f"digBAG       : {s['digit_cer_bag']:.4f}")
    print(f"miss/invented: {s['miss_rate']:.3f} / {s['invented_rate']:.3f}"
          f"  (invented vs GT+rapidocr: {s['invented_tokens']} tokens, "
          f"{s['invented_digits']} digit)")
    print(f"seconds/page : {s['seconds_per_page']:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Devanagari OCR bake-off")
    ap.add_argument("--model", choices=sorted(bakeoff_models.REGISTRY))
    ap.add_argument("--mode", choices=["lines", "pages"])
    ap.add_argument("--lines-source", default="heidata",
                    choices=["heidata", "nepali_lines"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--frozen", default=None, help="freeze manifest to verify")
    ap.add_argument("--json", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, kind, lic, ok, why in bakeoff_models.list_candidates():
            print(f"  {name:<12} {kind:<6} {'OK ' if ok else 'NO '} {lic[:44]:<46} {why}")
        return

    if not (args.model and args.mode):
        raise SystemExit("--model and --mode are required (or use --list)")

    cand = bakeoff_models.REGISTRY[args.model]
    ok, why = cand.available()
    if not ok:
        raise SystemExit(f"{args.model} unavailable: {why}")
    print(f"[bakeoff] {args.model} ({cand.kind}) | license: {cand.license}")

    if args.frozen:
        from eval_freeze import verify_manifest
        data_dir = args.data_dir
        if not data_dir:
            data_dir = (HEIDATA_DEFAULT if args.lines_source == "heidata" and
                        args.mode == "lines" else None)
        vok, problems = verify_manifest(args.frozen, data_dir=data_dir)
        if not vok:
            for p in problems[:10]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to score")
        print(f"[bakeoff] frozen verified: {os.path.basename(args.frozen)}")

    t0 = time.time()
    ctx = cand.load(lang=args.lang)
    load_s = time.time() - t0
    print(f"[bakeoff] model loaded in {load_s:.1f}s")

    if args.mode == "lines":
        if cand.kind == "pages":
            raise SystemExit(f"{args.model} cannot do lines")
        items = load_lines_source(args.lines_source, args.data_dir, args.limit)
        print(f"[bakeoff] {len(items)} line crops ({args.lines_source})")
        result = score_lines(cand, ctx, items, args.lang)
        print_lines(result["summary"])
    else:
        if cand.kind == "lines":
            raise SystemExit(f"{args.model} cannot do pages")
        if not args.data_dir:
            raise SystemExit("--data-dir is required for pages mode")
        result = score_pages(cand, ctx, args.data_dir, args.lang, args.limit)
        print_pages(result["summary"])

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        payload = {"model": args.model, "mode": args.mode,
                   "lines_source": args.lines_source if args.mode == "lines" else None,
                   "data_dir": args.data_dir, "lang": args.lang,
                   "load_seconds": round(load_s, 2),
                   "summary": result["summary"],
                   "worst" if args.mode == "lines" else "rows":
                       result.get("worst", result.get("rows"))}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[bakeoff] wrote {args.json}")


if __name__ == "__main__":
    main()
