"""
eval_document.py — document-mode evaluation harness.

Pipeline: page image -> method -> OCR backend -> metrics vs ground truth.

Methods (all numpy in/out, independently swappable):
    clean    : OCR on the undegraded page (ceiling / extraction floor)
    raw      : OCR on the degraded page (the baseline to beat)
    sauvola  : Sauvola binarization (classical baseline #2)
    lanczos2 : 2x Lanczos upscale (classical baseline #3)
    photo    : SmartUpscaler photo mode (proves photo SR helps or not; slow, opt-in)
    restore  : document_restore.restore_document (Phase B; skipped if absent)

Plus the pseudo-method `pipeline`: the actual shipped path
(document_pipeline.run_document_pipeline -> dual-stream OCR + digit-conflict
flags). It does its own OCR, so the backend loop only selects its backend.

Backends: rapidocr (default), tesseract (optional). Multiple can be compared.

Usage:
    python eval_document.py --build-synthetic 8 --levels mild,medium \
        --ocr rapidocr,tesseract --methods clean,raw,sauvola,lanczos2 \
        --json out/doc_smoke.json
    python eval_document.py --data-dir data/doc_eval/pdf --methods clean,raw,sauvola,restore
    python eval_document.py --build-devanagari 12 --script ne --lang ne \
        --methods clean,raw,pipeline --json out/doc_p4_ne.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import available_backends, ocr_page  # noqa: E402

Method = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def method_clean(img: np.ndarray) -> np.ndarray:
    return img


def method_raw(img: np.ndarray) -> np.ndarray:
    return img


def method_sauvola(img: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_sauvola
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = threshold_sauvola(gray, window_size=25, k=0.2)
    binary = (gray > t).astype(np.uint8) * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def method_lanczos2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)


def method_photo(img: np.ndarray) -> np.ndarray:
    from smart_upscaler import SmartUpscaler
    engine = method_photo._engine  # type: ignore[attr-defined]
    return engine.upscale(img, scale=2, mode="photo", fast=True, grain_strength=0.0)


method_photo._engine = None  # type: ignore[attr-defined]


def method_restore(img: np.ndarray) -> np.ndarray:
    try:
        from document_restore import restore_document
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"document_restore not available yet ({e})") from e
    out = restore_document(img, scale=1)
    return out["ocr_bgr"]


def _restore_stream(stream: str) -> Method:
    def fn(img: np.ndarray) -> np.ndarray:
        from document_restore import restore_document
        return restore_document(img, scale=1, ocr_stream=stream)["ocr_bgr"]
    return fn


METHODS: Dict[str, Method] = {
    "clean": method_clean,
    "raw": method_raw,
    "sauvola": method_sauvola,
    "lanczos2": method_lanczos2,
    "photo": method_photo,
    "restore": method_restore,
    "restore_clahe": _restore_stream("clahe"),
    "restore_sauvola": _restore_stream("sauvola"),
    "restore_gray": _restore_stream("gray"),
}


def _maybe_downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    if not max_side:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    f = max_side / m
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def run_evaluation(entries: List[Dict], methods: List[str], backends: List[str],
                   max_side: int = 0, with_photo: bool = False,
                   conf_threshold: float = 60.0, recheck_digits: bool = False,
                   repass_conf_below: float = 95.0,
                   lang: Optional[str] = None) -> Dict:
    results: Dict[str, Dict] = {}
    per_page: List[Dict] = []
    for i, entry in enumerate(entries, 1):
        gt = open(entry["_gt_path"], encoding="utf-8").read()
        degraded = cv2.imread(entry["_degraded_path"], cv2.IMREAD_COLOR)
        clean = cv2.imread(entry["_clean_path"], cv2.IMREAD_COLOR)
        if degraded is None:
            print(f"  [skip] missing {entry['_degraded_path']}")
            continue
        images = {"clean": clean if clean is not None else degraded, "raw": degraded}

        # pass 1: every method@backend OCRs the page
        ocr_results: Dict[str, object] = {}
        elapsed_by_key: Dict[str, float] = {}
        for method in methods:
            if method == "photo" and not with_photo:
                continue
            if method == "pipeline":
                from document_pipeline import run_document_pipeline
                for backend in backends:
                    t0 = time.time()
                    try:
                        pres = run_document_pipeline(
                            degraded, backend=backend, conf_threshold=conf_threshold,
                            repass_digits=recheck_digits, lang=lang)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [skip] pipeline@{backend}: {e}")
                        continue
                    key = f"pipeline@{backend}"
                    ocr_results[key] = pres.ocr
                    elapsed_by_key[key] = time.time() - t0
                continue
            try:
                if method in images:
                    img = images[method]
                else:
                    img = METHODS[method](degraded)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] method {method}: {e}")
                continue
            img = _maybe_downscale(img, max_side)
            for backend in backends:
                t0 = time.time()
                try:
                    res = ocr_page(img, backend=backend, conf_threshold=conf_threshold,
                                   recheck_digits=recheck_digits,
                                   repass_conf_below=repass_conf_below,
                                   lang=lang)
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {method}@{backend}: {e}")
                    continue
                key = f"{method}@{backend}"
                ocr_results[key] = res
                elapsed_by_key[key] = time.time() - t0

        # pass 2: score. On real sets (incomplete GT) peer methods' text counts
        # as evidence the token is real, so hallucination only counts tokens no
        # method produced and the annotator missed.
        peer_refs = bool(entry.get("real", False))
        for key, res in ocr_results.items():
            method, backend = key.split("@", 1)
            raw_res = ocr_results.get(f"raw@{backend}")
            raw_text = raw_res.text if raw_res is not None else ""
            others = [r.text for k, r in ocr_results.items()
                      if k != key and k.endswith("@" + backend)]
            stats = doc_metrics.token_stats(res.tokens, gt)
            hall = doc_metrics.hallucination_report(
                gt, raw_text, res.text, extra_refs=others if peer_refs else ())
            row = {
                "page": entry["id"], "method": method, "backend": backend,
                "cer": doc_metrics.cer(gt, res.text),
                "cer_bag": doc_metrics.cer_bag(gt, res.text),
                "digit_cer": doc_metrics.digit_cer(gt, res.text),
                "digit_cer_bag": doc_metrics.digit_cer(gt, res.text, bag=True),
                "wer": doc_metrics.wer(gt, res.text),
                "token_err": stats.token_error_rate,
                "coverage": stats.coverage,
                "false_alarm": stats.false_alarm_rate,
                "digit_errors": stats.digit_errors,
                "digit_flagged_errors": stats.digit_flagged_errors,
                "digit_flagged": stats.digit_flagged,
                "ece": doc_metrics.ece(res.tokens, gt),
                "tokens": stats.total,
                "flagged": stats.flagged,
                "invented": hall.get("invented", 0),
                "invented_digits": hall.get("invented_digits", 0),
                "repass_conflicts": res.meta.get("digit_repass_conflicts", 0),
                "seconds": round(elapsed_by_key[key], 3),
                "gt_chars": len(gt),
            }
            per_page.append(row)
            agg = results.setdefault(key, {"cer": [], "cer_bag": [],
                                           "digit_cer": [], "digit_cer_bag": [],
                                           "wer": [],
                                           "token_err": [],
                                           "coverage": [], "false_alarm": [],
                                           "ece": [], "seconds": [], "invented": 0,
                                           "invented_digits": 0, "repass_conflicts": 0,
                                           "digit_errors": 0, "digit_flagged_errors": 0,
                                           "digit_flagged": 0,
                                           "pages": 0})
            agg["cer"].append(row["cer"])
            agg["cer_bag"].append(row["cer_bag"])
            agg["digit_cer"].append(row["digit_cer"])
            agg["digit_cer_bag"].append(row["digit_cer_bag"])
            agg["wer"].append(row["wer"])
            agg["token_err"].append(row["token_err"])
            agg["coverage"].append(row["coverage"])
            agg["false_alarm"].append(row["false_alarm"])
            agg["ece"].append(row["ece"])
            agg["seconds"].append(row["seconds"])
            agg["invented"] += row["invented"]
            agg["invented_digits"] += row["invented_digits"]
            agg["repass_conflicts"] += row["repass_conflicts"]
            agg["digit_errors"] += row["digit_errors"]
            agg["digit_flagged_errors"] += row["digit_flagged_errors"]
            agg["digit_flagged"] += row["digit_flagged"]
            agg["pages"] += 1
        print(f"  [{i}/{len(entries)}] {entry['id']} done", flush=True)

    summary = {}
    for key, agg in results.items():
        summary[key] = {
            "pages": agg["pages"],
            "cer": float(np.mean(agg["cer"])) if agg["cer"] else None,
            "cer_bag": float(np.mean(agg["cer_bag"])) if agg["cer_bag"] else None,
            "digit_cer": float(np.mean(agg["digit_cer"])) if agg["digit_cer"] else None,
            "digit_cer_bag": (float(np.mean(agg["digit_cer_bag"]))
                              if agg["digit_cer_bag"] else None),
            "wer": float(np.mean(agg["wer"])) if agg["wer"] else None,
            "token_err": float(np.mean(agg["token_err"])) if agg["token_err"] else None,
            "coverage": float(np.mean(agg["coverage"])) if agg["coverage"] else None,
            "false_alarm": float(np.mean(agg["false_alarm"])) if agg["false_alarm"] else None,
            "ece": float(np.mean(agg["ece"])) if agg["ece"] else None,
            "seconds": float(np.mean(agg["seconds"])) if agg["seconds"] else None,
            "digit_coverage": (agg["digit_flagged_errors"] / agg["digit_errors"]
                               if agg["digit_errors"] else None),
            "digit_false_alarm": ((agg["digit_flagged"] - agg["digit_flagged_errors"])
                                  / agg["digit_flagged"] if agg["digit_flagged"] else None),
            "digit_errors": agg["digit_errors"],
            "digit_flagged": agg["digit_flagged"],
            "invented": agg["invented"],
            "invented_digits": agg["invented_digits"],
            "repass_conflicts": agg["repass_conflicts"],
        }
    return {"summary": summary, "per_page": per_page}


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def print_summary(summary: Dict) -> None:
    order = sorted(summary.items(), key=lambda kv: (kv[1]["cer"] if kv[1]["cer"] is not None else 9))
    header = (f"{'method@backend':<22}{'CER':>8}{'bagCER':>8}{'digCER':>8}{'digBAG':>8}"
              f"{'digCov':>8}{'digFA':>7}{'cover':>7}{'falseAl':>9}"
              f"{'ECE':>7}{'invent':>7}{'s/page':>8}")
    print("\n=== DOCUMENT EVAL (sorted by CER) ===")
    print(header)
    print("-" * len(header))
    for key, m in order:
        print(f"{key:<22}{m['cer']:>8.4f}{_fmt(m.get('cer_bag')):>8}{_fmt(m.get('digit_cer')):>8}"
              f"{_fmt(m.get('digit_cer_bag')):>8}{_fmt(m.get('digit_coverage')):>8}"
              f"{_fmt(m.get('digit_false_alarm')):>7}"
              f"{m['coverage']:>7.3f}{m['false_alarm']:>9.3f}{m['ece']:>7.3f}"
              f"{m['invented']:>7d}{m['seconds']:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description="Document restore evaluation harness")
    ap.add_argument("--data-dir", default=None, help="dir containing manifest.json")
    ap.add_argument("--build-synthetic", type=int, default=0,
                    help="build N synthetic pages under data/doc_eval/<layout> and evaluate")
    ap.add_argument("--layout", default="single", choices=["single", "two_column"],
                    help="synthetic page layout for --build-synthetic")
    ap.add_argument("--build-devanagari", type=int, default=0,
                    help="build N Devanagari fixture pages (Nepali/Hindi) and evaluate")
    ap.add_argument("--script", default="ne", choices=["ne", "hi"],
                    help="script for --build-devanagari")
    ap.add_argument("--dpi", type=int, default=300,
                    help="fixture render dpi for --build-devanagari")
    ap.add_argument("--levels", default="mild,medium")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--lang", default=None,
                    help="OCR language: en (default), ne/nepali, hi/hindi")
    ap.add_argument("--ocr", default=None, help="comma list; default all available")
    ap.add_argument("--methods", default="clean,raw,sauvola,lanczos2")
    ap.add_argument("--with-photo", action="store_true")
    ap.add_argument("--recheck-digits", action="store_true",
                    help="recognition-only re-pass on digit tokens (adds flags, never text)")
    ap.add_argument("--repass-conf-below", type=float, default=95.0,
                    help="re-pass digit tokens with conf below this (100 = all)")
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.build_synthetic:
        sub = "synthetic" if args.layout == "single" else f"synthetic_{args.layout}"
        out = os.path.join(BASE_DIR, "data", "doc_eval", sub)
        levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())
        doc_data.build_synthetic_dataset(out, n=args.build_synthetic,
                                         levels=levels, seed=args.seed,
                                         layout=args.layout)
        args.data_dir = out

    if args.build_devanagari:
        levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())
        sub = f"devanagari_{args.script}_{args.dpi}"
        out = os.path.join(BASE_DIR, "data", "doc_eval", sub)
        doc_data.build_devanagari_dataset(out, n=args.build_devanagari,
                                          script=args.script, levels=levels,
                                          dpi=args.dpi, seed=args.seed)
        args.data_dir = out

    if not args.data_dir:
        raise SystemExit("provide --data-dir or --build-synthetic N")

    manifest = doc_data.load_dataset(args.data_dir)
    entries = manifest["entries"]
    if args.pages:
        entries = entries[:args.pages]
    print(f"[doc-eval] {len(entries)} pages from {args.data_dir}")

    backends = [b.strip() for b in args.ocr.split(",")] if args.ocr else available_backends()
    backends = [b for b in backends if b in available_backends()]
    if not backends:
        raise SystemExit("no OCR backends available")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[doc-eval] methods={methods} backends={backends}")

    report = run_evaluation(entries, methods, backends, max_side=args.max_side,
                            with_photo=args.with_photo,
                            recheck_digits=args.recheck_digits,
                            repass_conf_below=args.repass_conf_below,
                            lang=args.lang)
    report["args"] = vars(args)
    report["dataset"] = {"dir": args.data_dir, "kind": manifest.get("kind"),
                         "pages": len(entries)}
    print_summary(report["summary"])

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[doc-eval] wrote {args.json}")


if __name__ == "__main__":
    main()
