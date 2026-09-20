"""
visual_compare.py — eyeball any LR image across multiple engines.

Builds a labeled panel: bicubic | engine A | engine B | ... and optional
256px center crops at full resolution.

Examples:
  python visual_compare.py --input De1.jpg --models x4plus,x4v3,ncnn:ultrasharp-4x
  python visual_compare.py --input testImg/foo1.jpg --models x4plus --crops
"""
from __future__ import annotations


# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from sr_engine import load_engine  # noqa: E402


def label_strip(width: int, text: str, height: int = 30) -> np.ndarray:
    strip = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(strip, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", default="x4plus,x4v3,ncnn:ultrasharp-4x")
    ap.add_argument("--out", default=None)
    ap.add_argument("--crops", action="store_true", help="also save 256px center crops at 1:1")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.input}")
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.out or os.path.join("out", f"{stem}_compare.png")

    panels = [(f"bicubic x4 ({w}x{h}->{w*4}x{h*4})",
               cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC))]
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        eng = load_engine(spec, 4, args.device)
        t0 = time.time()
        out = eng.upscale(img)
        dt = time.time() - t0
        print(f"{spec}: {out.shape[1]}x{out.shape[0]} in {dt:.2f}s")
        panels.append((f"{spec} ({dt:.1f}s)", out))
        eng.close()

    target_h = max(p.shape[0] for _lbl, p in panels)
    target_w = max(p.shape[1] for _lbl, p in panels)
    rows = []
    for lbl, p in panels:
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        canvas[:p.shape[0], :p.shape[1]] = p
        rows.append(np.vstack([label_strip(target_w, lbl), canvas]))
    sep = np.zeros((target_h + 30, 4, 3), dtype=np.uint8)
    interleaved = []
    for i, r in enumerate(rows):
        interleaved.append(r)
        if i < len(rows) - 1:
            interleaved.append(sep)
    sheet = np.hstack(interleaved)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    print(f"saved {out_path}")

    if args.crops:
        ch = min(256, h * 4)
        cw = min(256, w * 4)
        crop_rows = []
        for lbl, p in panels:
            cy, cx = p.shape[0] // 2, p.shape[1] // 2
            crop = p[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2]
            crop_rows.append(np.vstack([label_strip(crop.shape[1], lbl), crop]))
        sep = np.zeros((crop_rows[0].shape[0], 4, 3), dtype=np.uint8)
        crop_sheet = np.hstack([x for pair in zip(crop_rows, [sep] * len(crop_rows)) for x in pair][:-1])
        crop_path = out_path.replace(".png", "_crop.png")
        cv2.imwrite(crop_path, crop_sheet)
        print(f"saved {crop_path}")


if __name__ == "__main__":
    main()
