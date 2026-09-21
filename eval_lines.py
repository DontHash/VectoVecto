"""
eval_lines.py — recognition-only evaluation on real text-line crops.

Why a separate harness: page metrics conflate detection and recognition. These
sets are cropped lines with transcriptions, so timing/detection bugs cannot
hide a recognizer regression (and vice versa). The scoring function takes a
recognizer callable so tests can use a fake one.

Sets this is built for (see data/doc_eval):
  * nepali_lines  — himalaya-ai/nepali-deva-ocr-eval (500 real Nepali print
    lines; provenance/license UNKNOWN, internal eval only, human eyeball pass
    required before its numbers may be quoted)

Usage:
    python eval_lines.py --data-dir data/doc_eval/nepali_lines --lang ne \
        --bootstrap 2000 --json out/nepali_lines.json
    python eval_lines.py --data-dir data/doc_eval/nepali_lines --lang ne \
        --contact-sheet out/nepali_lines_sheet.png --sheet-n 25
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import available_backends, get_backend, normalize_lang  # noqa: E402

Recognizer = Callable[[np.ndarray], Tuple[str, float]]


def score_lines(entries: List[Dict], recognize: Recognizer,
                limit: int = 0) -> Dict:
    """Score each line crop with `recognize` (image_bgr -> text, conf)."""
    rows: List[Dict] = []
    for e in entries:
        if limit and len(rows) >= limit:
            break
        img = cv2.imread(e["_image_path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] missing {e['_image_path']}")
            continue
        gt = open(e["_gt_path"], encoding="utf-8").read()
        t0 = time.time()
        hyp, conf = recognize(img)
        rows.append({
            "id": e["id"],
            "gt": gt,
            "hyp": hyp,
            "conf": float(conf),
            "cer": doc_metrics.cer(gt, hyp),
            "wer": doc_metrics.wer(gt, hyp),
            "digit_cer": doc_metrics.digit_cer(gt, hyp),
            "seconds": round(time.time() - t0, 4),
            "has_digits": bool(doc_metrics.digit_tokens(gt)),
        })
    cers = [r["cer"] for r in rows]
    wers = [r["wer"] for r in rows]
    digit_rows = [r for r in rows if r["has_digits"]]
    exact = [1.0 if r["cer"] == 0.0 else 0.0 for r in rows]
    summary = {
        "lines": len(rows),
        "cer": float(np.mean(cers)) if cers else None,
        "wer": float(np.mean(wers)) if wers else None,
        "exact_match": float(np.mean(exact)) if exact else None,
        "cer_median": float(statistics.median(cers)) if cers else None,
        "digit_lines": len(digit_rows),
        "digit_cer": (float(np.mean([r["digit_cer"] for r in digit_rows]))
                      if digit_rows else None),
        "seconds_per_line": (float(np.mean([r["seconds"] for r in rows]))
                             if rows else None),
        "empty_outputs": sum(1 for r in rows if not r["hyp"].strip()),
    }
    if rows:
        cers_sorted = np.asarray(cers, dtype=float)
        summary["cer_ci"] = [round(v, 4) for v in
                             doc_metrics.bootstrap_ci(cers_sorted, n_boot=2000)]
    worst = sorted(rows, key=lambda r: -r["cer"])[:20]
    return {"summary": summary, "rows": rows, "worst": worst}


def make_contact_sheet(rows: List[Dict], out_path: str, n: int = 25,
                       scale_h: int = 46, width: int = 1300) -> str:
    """Render crops + GT (+hypothesis when it differs) for human verification."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for path, idx in (("C:/Windows/Fonts/Nirmala.ttc", 0),
                      ("C:/Windows/Fonts/arial.ttf", 0)):
        try:
            font = ImageFont.truetype(path, 18, index=idx)
            break
        except Exception:  # noqa: BLE001
            continue

    picked = rows[:n]
    row_h = scale_h + 46
    sheet = Image.new("RGB", (width, row_h * len(picked) + 10), "white")
    draw = ImageDraw.Draw(sheet)
    y = 4
    for r in picked:
        img = cv2.imread(r["_image_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        f = scale_h / max(1, h)
        im = cv2.resize(img, (max(1, int(w * f)), scale_h),
                        interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sheet.paste(pil, (6, y))
        label_y = y + scale_h + 2
        gt = r["gt"].strip().replace("\n", " ")
        hyp = r["hyp"].strip().replace("\n", " ")
        draw.text((6, label_y), f"GT : {gt[:70]}", fill=(0, 0, 0), font=font)
        color = (0, 128, 0) if r["cer"] == 0 else (180, 0, 0)
        draw.text((6, label_y + 21), f"OCR: {hyp[:70]}  (CER {r['cer']:.2f})",
                  fill=color, font=font)
        y += row_h
    Image.fromarray(np.array(sheet)).save(out_path)
    return out_path


def write_review_csv(rows: List[Dict], out_path: str) -> str:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "cer", "gt", "ocr"])
        for r in sorted(rows, key=lambda x: -x["cer"]):
            w.writerow([r["id"], f"{r['cer']:.4f}", r["gt"], r["hyp"]])
    return out_path


def print_report(result: Dict) -> None:
    s = result["summary"]
    print("\n=== LINE-CROP EVAL (recognition only) ===")
    print(f"lines          : {s['lines']}")
    print(f"CER            : {s['cer']:.4f}" if s["cer"] is not None else "CER: n/a")
    if s.get("cer_ci"):
        print(f"CER 95% CI     : [{s['cer_ci'][0]:.3f} - {s['cer_ci'][1]:.3f}]")
    print(f"CER median     : {s['cer_median']:.4f}")
    print(f"WER            : {s['wer']:.4f}")
    print(f"exact match    : {s['exact_match']:.3f}")
    if s["digit_cer"] is not None:
        print(f"digit-line CER : {s['digit_cer']:.4f} ({s['digit_lines']} lines)")
    print(f"empty outputs  : {s['empty_outputs']}")
    print(f"s/line         : {s['seconds_per_line']:.3f}")
    print("\nworst 5:")
    for r in result["worst"][:5]:
        print(f"  {r['id']} CER {r['cer']:.2f}  GT={r['gt'][:40]!r} HYP={r['hyp'][:40]!r}")


def main():
    ap = argparse.ArgumentParser(description="Recognition-only line-crop eval")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--build-from", default=None,
                    help="HF repo to build the line dataset from first")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--lang", default=None, help="en (default), ne, hi")
    ap.add_argument("--ocr", default="rapidocr")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap resamples for the CER CI (e.g. 2000)")
    ap.add_argument("--frozen", default=None,
                    help="freeze manifest to verify before scoring")
    ap.add_argument("--contact-sheet", default=None,
                    help="write a GT-vs-OCR contact sheet for human review")
    ap.add_argument("--sheet-n", type=int, default=25)
    ap.add_argument("--review-csv", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.build_from:
        doc_data.build_line_crop_dataset(args.build_from, args.data_dir,
                                         max_lines=args.max_lines)
        print(f"[lines] built {args.data_dir}")

    if args.frozen:
        from eval_freeze import freeze_info, verify_manifest
        ok, problems = verify_manifest(args.frozen, data_dir=args.data_dir)
        if not ok:
            for p in problems[:20]:
                print(f"  [frozen] DRIFT: {p}")
            raise SystemExit("frozen dataset drift — refusing to score")
        print(f"[lines] frozen verified: {freeze_info(args.frozen)['name']}")

    manifest = doc_data.load_line_dataset(args.data_dir)
    entries = manifest["entries"]
    if args.limit:
        entries = entries[:args.limit]
    print(f"[lines] {len(entries)} crops from {args.data_dir} "
          f"(lang={normalize_lang(args.lang)})")

    if args.ocr not in available_backends():
        raise SystemExit(f"OCR backend {args.ocr!r} unavailable")
    backend = get_backend(args.ocr)

    def recognize(img: np.ndarray) -> Tuple[str, float]:
        return backend.recognize_crop(img, lang=args.lang)

    result = score_lines(entries, recognize)
    print_report(result)
    if args.bootstrap:
        vals = np.asarray([r["cer"] for r in result["rows"]], dtype=float)
        ci = doc_metrics.bootstrap_ci(vals, n_boot=args.bootstrap, seed=1)
        result["summary"]["cer_ci_bootstrap"] = [round(ci[0], 4), round(ci[1], 4)]
        print(f"CER CI ({args.bootstrap} resamples): "
              f"[{ci[0]:.3f} - {ci[1]:.3f}]")

    if args.contact_sheet:
        # contact sheets need the image paths on the rows
        by_id = {e["id"]: e for e in entries}
        for r in result["rows"]:
            r["_image_path"] = by_id[r["id"]]["_image_path"]
        make_contact_sheet(result["rows"], args.contact_sheet, n=args.sheet_n)
        print(f"[lines] contact sheet -> {args.contact_sheet}")

    if args.review_csv:
        write_review_csv(result["rows"], args.review_csv)
        print(f"[lines] review csv -> {args.review_csv}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        out = {"summary": result["summary"], "worst": result["worst"],
               "dataset": {"dir": args.data_dir, "source": manifest.get("source"),
                           "split": manifest.get("split"),
                           "lines": len(entries)},
               "args": {k: v for k, v in vars(args).items()}}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[lines] wrote {args.json}")


if __name__ == "__main__":
    main()
