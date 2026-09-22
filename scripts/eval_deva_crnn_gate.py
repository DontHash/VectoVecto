"""
eval_deva_crnn_gate.py — run the W1 adopt-if gate with a trained checkpoint.

Usage:
    python scripts/eval_deva_crnn_gate.py --ckpt out/deva_crnn/ckpt.pt \
        --json out/deva_crnn_gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from deva_crnn.gate import (DIGIT_EXACT_BAR, heidata_line_crops,  # noqa: E402
                            line_stats, v2_anchor_pages)
from deva_crnn.predict import load_model, recognize_lines  # noqa: E402


def _run_lines(model, charset, crops, batch=32, decode_mode="greedy",
               beam_width=8):
    hyps = []
    for i in range(0, len(crops), batch):
        hyps.extend(recognize_lines(model, charset, crops[i:i + batch],
                                    decode_mode=decode_mode,
                                    beam_width=beam_width))
    return hyps


def main():
    ap = argparse.ArgumentParser(description="W1 gate evaluation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data",
                                                       "doc_eval",
                                                       "heidata_printed"))
    ap.add_argument("--v2-dir", default=os.path.join(BASE_DIR, "data",
                                                     "doc_eval",
                                                     "nepali_pdf_v2"))
    ap.add_argument("--readings", default=os.path.join(BASE_DIR, "out",
                                                       "anchor_gemini",
                                                       "gemini_readings.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decode", choices=("greedy", "beam"), default="greedy")
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    model, charset = load_model(args.ckpt, "cpu")
    result = {"decode": args.decode,
              "beam_width": args.beam_width if args.decode == "beam" else None}

    t0 = time.time()
    gts, crops = heidata_line_crops(args.data_dir, limit=args.limit)
    hyps = _run_lines(model, charset, crops, decode_mode=args.decode,
                      beam_width=args.beam_width)
    result["heidata_lines"] = {**line_stats(gts, hyps),
                               "seconds": round(time.time() - t0, 1)}

    t0 = time.time()
    pages = v2_anchor_pages(args.v2_dir, args.readings, limit=args.limit)
    page_rows = []
    for p in pages:
        line_hyps = _run_lines(model, charset, p["crops"],
                               decode_mode=args.decode,
                               beam_width=args.beam_width)
        hyp_page = "\n".join(line_hyps)
        stats = line_stats([p["gt"]], [hyp_page])
        stats["page"] = p["page"]
        page_rows.append(stats)
    result["v2_anchor_pages"] = {
        "pages": len(page_rows),
        "digit_exact_pages": (sum(1 for r in page_rows
                                  if r["digit_exact"] == 1.0)
                              / len(page_rows)) if page_rows else None,
        "bagcer": (sum(r["bagcer"] for r in page_rows) / len(page_rows))
        if page_rows else None,
        "seconds": round(time.time() - t0, 1),
        "rows": page_rows,
    }

    heid = result["heidata_lines"]
    result["gate"] = {
        "digit_exact_bar": DIGIT_EXACT_BAR,
        "heidata_digit_exact": heid["digit_exact"],
        "heidata_pass": (heid["digit_exact"] is not None
                         and heid["digit_exact"] >= DIGIT_EXACT_BAR),
        "v2_digit_exact_pages": result["v2_anchor_pages"]["digit_exact_pages"],
    }
    result["gate"]["pass"] = bool(result["gate"]["heidata_pass"])

    print("\n=== W1 GATE (deva_crnn) ===")
    print("heiDATA lines :", {k: v for k, v in heid.items()})
    print("v2 anchor     :", {k: v for k, v in result["v2_anchor_pages"].items()
                              if k != "rows"})
    print("gate          :", result["gate"])
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
