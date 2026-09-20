"""
cli.py — batch folder super-resolution CLI.

Examples:
  python cli.py --input photos --output photos_x4 --recursive --skip-existing
  python cli.py --input shot.jpg --output out --model auto --tta
  python cli.py --input icons --output icons_x4 --model ncnn:ultrasharp-4x --format png --report run.json
  python cli.py --input catalog --output catalog_x4 --format webp --quality 95 --suffix _hd

Notes:
  * Alpha channels are preserved (RGB upscaled by the model, alpha by Lanczos).
  * `--workers` >1 is only used for CPU-side engines (ncnn/classical); GPU
    engines (torch/ONNX-DirectML) run single-process to avoid device contention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sr_engine import IMG_EXTS, Engine, load_engine  # noqa: E402


def list_inputs(path: str, recursive: bool) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    if recursive:
        for dirpath, _dirs, names in os.walk(path):
            for n in names:
                if n.lower().endswith(IMG_EXTS):
                    files.append(os.path.join(dirpath, n))
    else:
        for n in os.listdir(path):
            p = os.path.join(path, n)
            if os.path.isfile(p) and n.lower().endswith(IMG_EXTS):
                files.append(p)
    return sorted(files)


def split_alpha(img: np.ndarray):
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    return img, None


def merge_alpha(rgb: np.ndarray, alpha_small: Optional[np.ndarray],
                scale: int) -> np.ndarray:
    if alpha_small is None:
        return rgb
    alpha = cv2.resize(alpha_small, (rgb.shape[1], rgb.shape[0]),
                       interpolation=cv2.INTER_LANCZOS4)
    return np.dstack([rgb, alpha])


def out_name(src: str, out_dir: str, suffix: str, ext: str, flat: bool,
             base_in: str) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    if flat:
        sub = out_dir
    else:
        rel = os.path.relpath(os.path.dirname(os.path.abspath(src)),
                              os.path.abspath(base_in))
        sub = os.path.join(out_dir, rel) if rel != "." else out_dir
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{stem}{suffix}.{ext}")


def main():
    ap = argparse.ArgumentParser(description="Batch SR folder upscaler")
    ap.add_argument("--input", required=True, help="file or folder")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--model", default="auto",
                    help="auto | <path>.pth | <path>.onnx | ncnn:<name> | bicubic | lanczos")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--flat", action="store_true", help="do not mirror subfolders")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--format", choices=["png", "webp", "jpg"], default="png")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--grain", type=float, default=0.0,
                    help="optional luminance micro-grain (0 = off, professional default)")
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 only for ncnn/classical engines")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")
    os.makedirs(args.output, exist_ok=True)

    files = list_inputs(args.input, args.recursive)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no input images found")
    base_in = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)

    print(f"[cli] {len(files)} files | model={args.model} scale={args.scale} format={args.format}")
    t0 = time.time()
    engine: Engine = load_engine(args.model, args.scale, args.device, args.tile,
                                 fp16=not args.no_fp16, tta=args.tta)
    print(f"[cli] engine: {engine}")

    records: List[Dict] = []
    ok, skipped, failed = 0, 0, 0
    for i, src in enumerate(files, 1):
        dst = out_name(src, args.output, args.suffix, args.format, args.flat, base_in)
        if args.skip_existing and os.path.exists(dst):
            skipped += 1
            records.append({"src": src, "dst": dst, "status": "skipped"})
            continue
        try:
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("unreadable")
            bgr, alpha = split_alpha(img)
            t1 = time.time()
            sr = engine.upscale(bgr)
            dt = time.time() - t1
            sr = merge_alpha(sr, alpha, args.scale)
            if args.grain and args.grain > 0:
                noise = np.random.normal(0, args.grain, sr[:, :, :3].shape)
                sr[:, :, :3] = np.clip(sr[:, :, :3].astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
            if args.format == "png":
                cv2.imwrite(dst, sr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            elif args.format == "webp":
                cv2.imwrite(dst, sr, [cv2.IMWRITE_WEBP_QUALITY, args.quality])
            else:
                cv2.imwrite(dst, sr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            ok += 1
            records.append({"src": src, "dst": dst, "status": "ok",
                            "seconds": round(dt, 3),
                            "in": f"{img.shape[1]}x{img.shape[0]}",
                            "out": f"{sr.shape[1]}x{sr.shape[0]}"})
            print(f"  [{i}/{len(files)}] {os.path.basename(src)} -> {sr.shape[1]}x{sr.shape[0]} ({dt:.2f}s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            records.append({"src": src, "dst": dst, "status": "error", "error": str(e)})
            print(f"  [{i}/{len(files)}] FAILED {src}: {e}")

    total = time.time() - t0
    summary = {"files": len(files), "ok": ok, "skipped": skipped, "failed": failed,
               "seconds": round(total, 2), "engine": repr(engine), "args": vars(args)}
    print(f"[cli] done: {ok} ok, {skipped} skipped, {failed} failed in {total:.1f}s")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"[cli] report: {args.report}")
    engine.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
