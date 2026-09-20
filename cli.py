"""
cli.py — batch super-resolution / document-restore CLI.

Photo mode (default):
  python cli.py --input photos --output photos_x4 --recursive --skip-existing
  python cli.py --input shot.jpg --output out --model auto --tta
  python cli.py --input icons --output icons_x4 --model ncnn:ultrasharp-4x --format png --report run.json
  python cli.py --input catalog --output catalog_x4 --format webp --quality 95 --suffix _hd

Document mode (searchable PDF + overlay + transcript, local):
  python cli.py --mode document --input scan.jpg --output out_dir
  python cli.py --mode document --input pack.pdf --output out_dir --max-pages 5 --ocr rapidocr
  python cli.py --mode document --input photos_dir --output out_dir --recursive --skip-existing

Notes:
  * Alpha channels are preserved (RGB upscaled by the model, alpha by Lanczos).
  * `--workers` >1 is only used for CPU-side engines (ncnn/classical); GPU
    engines (torch/ONNX-DirectML) run single-process to avoid device contention.
  * Document mode never needs the cloud: OCR models are bundled with RapidOCR.
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


# ---------------------------------------------------------------------------
# document mode
# ---------------------------------------------------------------------------

DOC_EXTS = IMG_EXTS + (".pdf",)


def _list_doc_inputs(path: str, recursive: bool) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    if recursive:
        for dirpath, _dirs, names in os.walk(path):
            for n in names:
                if n.lower().endswith(DOC_EXTS):
                    files.append(os.path.join(dirpath, n))
    else:
        for n in os.listdir(path):
            p = os.path.join(path, n)
            if os.path.isfile(p) and n.lower().endswith(DOC_EXTS):
                files.append(p)
    return sorted(files)


def run_document_mode(args) -> int:
    import doc_data
    from document_pipeline import pick_backend, run_document_pipeline

    try:
        backend = pick_backend(args.ocr)
    except RuntimeError as e:
        raise SystemExit(str(e))

    files = _list_doc_inputs(args.input, args.recursive)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no input documents found")

    dpi = args.dpi or None
    records: List[Dict] = []
    ok = skipped = failed = 0
    t_start = time.time()
    print(f"[cli] document mode | {len(files)} inputs | ocr={backend} "
          f"deskew={args.deskew} pdf={not args.no_pdf}")

    def process(img: np.ndarray, name: str, src: str):
        nonlocal ok, failed
        pdf_path = os.path.join(args.output, f"{name}.pdf")
        if args.skip_existing and os.path.exists(pdf_path):
            records.append({"src": src, "name": name, "status": "skipped"})
            return "skipped"
        os.makedirs(args.output, exist_ok=True)
        try:
            res = run_document_pipeline(
                img, backend=backend, lang=args.lang, deskew=args.deskew,
                repass_digits=args.repass_digits, dpi=dpi,
                reading_order=not getattr(args, "no_reading_order", False),
                auto_rotate=getattr(args, "rotate", "auto") != "off",
                out_dir=args.output, stem=name,
                make_pdf=not args.no_pdf, make_overlay=not args.no_overlay,
                make_txt=not args.no_txt, make_json=True)
            ok += 1
            low = sum(1 for t in res.ocr.tokens if "low_conf" in t.flags)
            review = res.review_list
            records.append({"src": src, "name": name, "status": "ok",
                            "tokens": len(res.ocr.tokens), "low_conf": low,
                            "digit_conflicts": res.meta["digit_conflicts"],
                            "review": [{"text": t.text, "flags": t.flags}
                                       for t in review[:5]],
                            "seconds": res.meta["seconds"],
                            "skew_angle": res.meta["skew_angle"],
                            "outputs": res.outputs})
            print(f"  {name}: {res.status_line}")
            for tok in review[:3]:
                print(f"    review: {tok.text!r} ({', '.join(tok.flags)})")
            return "ok"
        except Exception as e:  # noqa: BLE001
            failed += 1
            records.append({"src": src, "name": name, "status": "error", "error": str(e)})
            print(f"  FAILED {name}: {e}")
            return "error"

    for src in files:
        try:
            if src.lower().endswith(".pdf"):
                pages = list(doc_data.pdf_to_pages(src, dpi=args.dpi or 200))
                if args.max_pages:
                    pages = pages[:args.max_pages]
                stem = os.path.splitext(os.path.basename(src))[0]
                for idx, page_img, _gt in pages:
                    process(page_img, f"{stem}_p{idx:03d}", f"{src}#p{idx}")
            else:
                from document_orientation import load_image_bgr
                img = load_image_bgr(src)
                if img is None:
                    raise ValueError("unreadable image")
                stem = os.path.splitext(os.path.basename(src))[0]
                status = process(img, stem, src)
                if status == "skipped":
                    skipped += 1
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            failed += 1
            records.append({"src": src, "status": "error", "error": str(e)})
            print(f"  FAILED {src}: {e}")

    total = time.time() - t_start
    print(f"[cli] document mode done: {ok} ok, {skipped} skipped, {failed} failed "
          f"in {total:.1f}s")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": {"inputs": len(files), "ok": ok, "skipped": skipped,
                                   "failed": failed, "seconds": round(total, 2),
                                   "ocr": backend}, "records": records}, f, indent=2)
        print(f"[cli] report: {args.report}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description="Batch SR upscaler / document restore")
    ap.add_argument("--mode", choices=["photo", "document"], default="photo")
    ap.add_argument("--input", required=True, help="file or folder (document mode also accepts .pdf)")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--model", default="auto",
                    help="photo mode: auto | <path>.pth | <path>.onnx | ncnn:<name> | bicubic | lanczos")
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
    doc = ap.add_argument_group("document mode")
    doc.add_argument("--ocr", default=None, help="rapidocr | tesseract (default: best available)")
    doc.add_argument("--lang", default=None, help="OCR language (backend dependent)")
    doc.add_argument("--deskew", action="store_true",
                     help="rotate to deskew; OCR then runs on the display for alignment")
    doc.add_argument("--repass-digits", action="store_true", dest="repass_digits",
                     help="re-read digit tokens for the review queue; off until "
                          "re-measured with the engine fix (Appendix A.1)")
    doc.add_argument("--max-pages", type=int, default=1, dest="max_pages",
                     help="max pages per PDF input (default 1)")
    doc.add_argument("--no-reading-order", action="store_true", dest="no_reading_order",
                     help="skip XY-cut reading-order repair for column layouts")
    doc.add_argument("--rotate", choices=["auto", "off"], default="auto",
                     help="auto: EXIF + line-classifier votes/geometry (all four "
                          "angles, all page sizes); off = flag only")
    doc.add_argument("--dpi", type=int, default=0, help="source DPI for PDF page size (0 = auto)")
    doc.add_argument("--no-pdf", action="store_true", dest="no_pdf")
    doc.add_argument("--no-overlay", action="store_true", dest="no_overlay")
    doc.add_argument("--no-txt", action="store_true", dest="no_txt")
    args = ap.parse_args()

    if args.mode == "document":
        sys.exit(run_document_mode(args))

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
