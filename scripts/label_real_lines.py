"""
label_real_lines.py — real Devanagari line crops with Gemini line labels (W1.2).

Attempt 1 trained on synthetic lines only and failed the gate: the model
memorized the renderer. This script mines *real* lines from the v2 source PDFs
- pages that are NOT part of the frozen anchor - and labels them with Gemini
(page transcription, same prompt/model as the anchor), aligned to RapidOCR
line boxes with a monotone DP matcher. Only high-similarity alignments are
kept, so a Gemini slip becomes a dropped line, not a wrong label.

Provenance: labels are model-produced (gemini-2.5-pro), not human GT. The
frozen gate stays untouched by construction (frozen pages are skipped).

Usage:
    python scripts/label_real_lines.py --out data/doc_eval/deva_real_lines \
        --per-doc 12 --json out/real_lines_report.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from anchor_gemini import DEFAULT_MODEL, transcribe_page  # noqa: E402

MIN_SIM = 0.60
MIN_LEN = 3
MAX_LEN = 48
_DEVA_LO, _DEVA_HI = "\u0900", "\u097f"
MONTAGE_ROWS = 16
MIN_DEVA_SHARE = 0.2

MONTAGE_PROMPT = (
    "This image stacks {n} horizontal strips, each one printed line of a "
    "Devanagari document. Transcribe every strip, top to bottom, exactly as "
    "printed (keep Devanagari digits, do not translate). Output exactly one "
    "line per strip, in order, plain text only - no numbering, no commentary."
)


def build_montage(crops: List[np.ndarray], gap: int = 10,
                  pad: int = 8) -> np.ndarray:
    """Stack line crops into one image: strip i is crop i (top to bottom)."""
    w = max(c.shape[1] for c in crops) + 2 * pad
    h = sum(c.shape[0] for c in crops) + gap * (len(crops) - 1) + 2 * pad
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    y = pad
    for c in crops:
        canvas[y:y + c.shape[0], pad:pad + c.shape[1]] = c
        y += c.shape[0] + gap
    return canvas


def _call_gemini(client, prompt: str, img: np.ndarray, model: str,
                 timeout_s: float = 90.0, retries: int = 2) -> str:
    import time as _time

    from PIL import Image
    config = None
    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.0,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    except Exception:  # noqa: BLE001 - fake clients in tests
        config = None
    contents = [prompt, Image.fromarray(img[:, :, ::-1])]
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model,
                                                  contents=contents,
                                                  config=config)
            return (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                _time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"gemini call failed after {retries + 1} attempts: "
                       f"{last_err}")


def transcribe_montage(client, crops: List[np.ndarray],
                       hints: Optional[List[str]] = None,
                       model: str = DEFAULT_MODEL, rows: int = MONTAGE_ROWS,
                       timeout_s: float = 90.0, retries: int = 2) -> List[str]:
    """One Gemini call per `rows` crops; returns one text per crop (or "").

    The montage fixes segmentation (strip i IS crop i), so the happy path is a
    1:1 line count. If Gemini merged or split strips anyway, fall back to a DP
    alignment against the OCR text of each crop (`hints`) and keep only the
    confident pairs - a dropped line beats a wrong label.
    """
    texts: List[str] = [""] * len(crops)
    for start in range(0, len(crops), rows):
        chunk = crops[start:start + rows]
        prompt = MONTAGE_PROMPT.format(n=len(chunk))
        raw = _call_gemini(client, prompt, build_montage(chunk), model,
                           timeout_s=timeout_s, retries=retries)
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if len(lines) == len(chunk):
            texts[start:start + len(chunk)] = lines
            continue
        chunk_hints = (hints or [])[start:start + len(chunk)]
        if not chunk_hints:
            continue
        for i, j, _s in align_lines(chunk_hints, lines, min_sim=0.35):
            texts[start + i] = lines[j]
    return texts


def _is_deva(text: str) -> bool:
    return any(_DEVA_LO <= c <= _DEVA_HI for c in text)


def _is_labelable(text: str) -> bool:
    """Devanagari text or bare numbers - numeric cells train the digit read.

    Anything outside the training charset is filtered later, at export time.
    """
    return _is_deva(text) or any(c.isdigit() for c in text)


def align_lines(ocr_texts: List[str], gemini_lines: List[str],
                min_sim: float = MIN_SIM) -> List[Tuple[int, int, float]]:
    """Monotone alignment between OCR lines and Gemini lines.

    Both sequences are in reading order, so a Needleman-Wunsch-style DP over
    normalized strings beats per-line best-match: it cannot cross-match.
    Returns (ocr_index, gemini_index, similarity) triples.
    """
    a = [doc_metrics.normalize_text(t) for t in ocr_texts]
    b = [doc_metrics.normalize_text(t) for t in gemini_lines]
    n, m = len(a), len(b)
    if not n or not m:
        return []
    sim = [[difflib.SequenceMatcher(None, a[i], b[j], autojunk=False).ratio()
            for j in range(m)] for i in range(n)]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            best = row[j - 1] if row[j - 1] > prev[j] else prev[j]
            s = sim[i - 1][j - 1]
            if s >= min_sim and prev[j - 1] + s > best:
                best = prev[j - 1] + s
            row[j] = best
    pairs: List[Tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 and j > 0:
        s = sim[i - 1][j - 1]
        if s >= min_sim and abs(dp[i][j] - (dp[i - 1][j - 1] + s)) < 1e-9:
            pairs.append((i - 1, j - 1, s))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def frozen_pages(manifest_path: str) -> Dict[str, set]:
    """Source basename -> set of page indices used by the frozen anchor."""
    data = json.load(open(manifest_path, encoding="utf-8"))
    used: Dict[str, set] = {}
    for e in data["entries"]:
        used.setdefault(os.path.basename(e["source"]), set()).add(e["page"])
    return used


def pick_pages(v2_dir: str, per_doc: int, max_pages: int = 0,
               dpi: int = 300) -> List[Tuple[str, List[int]]]:
    """(source_path, page_indices) pairs: pages after the frozen ones."""
    frozen = frozen_pages(os.path.join(v2_dir, "manifest.json"))
    plan: List[Tuple[str, List[int]]] = []
    total = 0
    for src, used in sorted(frozen.items()):
        path = None
        for alt in ("nepali_pdf_sources_v2", "nepali_pdf_sources"):
            cand = os.path.join(os.path.dirname(v2_dir), alt, src)
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            print(f"  [skip] source not found: {src}")
            continue
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(path)
        n = len(doc)
        doc.close()
        start = max(used) + 1
        pages = list(range(start, min(n, start + per_doc)))
        if not pages:
            continue
        plan.append((path, pages))
        total += len(pages)
        if max_pages and total >= max_pages:
            break
    return plan


def label_page(client, img: np.ndarray, page_id: str, model: str,
               strips: int, min_sim: float, ocr_fn=None,
               transcribe_fn=None) -> Dict:
    """Align one page: returns kept (crop, text, sim) rows + stats."""
    ocr_fn = ocr_fn or _default_ocr
    transcribe_fn = transcribe_fn or _default_transcribe
    rec = ocr_fn(img)
    gemini_text = transcribe_fn(client, img, model=model, strips=strips)
    gemini_lines = [ln for ln in gemini_text.split("\n") if ln.strip()]
    ocr_texts = [t.text for t in rec.tokens]
    pairs = align_lines(ocr_texts, gemini_lines, min_sim=min_sim)
    kept, dropped, seen = [], 0, set()
    for i, j, s in pairs:
        text = doc_metrics.normalize_text(gemini_lines[j])
        if not (MIN_LEN <= len(text) <= MAX_LEN) or not _is_deva(text):
            dropped += 1
            continue
        if text in seen:  # strip-overlap duplicate: one label per text
            dropped += 1
            continue
        x0, y0, x1, y1 = rec.tokens[i].bbox
        pad = 4
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        if crop.size == 0:
            dropped += 1
            continue
        seen.add(text)
        kept.append({"crop": crop, "text": text, "sim": round(s, 4),
                     "box": [int(x0), int(y0), int(x1), int(y1)]})
    return {"page": page_id, "gemini_text": gemini_text, "ocr_lines": len(ocr_texts),
            "gemini_lines": len(gemini_lines), "matched": len(pairs),
            "kept": kept, "dropped": dropped}


def _default_ocr(img: np.ndarray):
    from document_ocr import ocr_page
    return ocr_page(img, backend="rapidocr", lang="ne")


def label_page_montage(client, img: np.ndarray, page_id: str, model: str,
                       rows: int = MONTAGE_ROWS, ocr_fn=None,
                       montage_fn=None) -> Dict:
    """Label the page's OCR line crops directly (montage, 1:1 by design)."""
    ocr_fn = ocr_fn or _default_ocr
    montage_fn = montage_fn or transcribe_montage
    rec = ocr_fn(img)
    crops, hints, boxes = [], [], []
    for t in rec.tokens:
        x0, y0, x1, y1 = t.bbox
        pad = 4
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        if crop.size == 0:
            continue
        crops.append(crop)
        hints.append(t.text)
        boxes.append([int(x0), int(y0), int(x1), int(y1)])
    texts = montage_fn(client, crops, hints=hints, model=model, rows=rows)
    kept, dropped = [], {"empty": 0, "length": 0, "chars": 0}
    for crop, text, hint, box in zip(crops, texts, hints, boxes):
        text = doc_metrics.normalize_text(text)
        if not text:
            dropped["empty"] += 1
            continue
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            dropped["length"] += 1
            continue
        if not _is_labelable(text):
            dropped["chars"] += 1
            continue
        sim = difflib.SequenceMatcher(None, doc_metrics.normalize_text(hint),
                                      text, autojunk=False).ratio()
        kept.append({"crop": crop, "text": text, "sim": round(sim, 4),
                     "box": box})
    return {"page": page_id, "gemini_text": "", "ocr_lines": len(crops),
            "gemini_lines": len([t for t in texts if t]), "matched": len(crops),
            "kept": kept, "dropped": sum(dropped.values()),
            "dropped_reasons": dropped}


def _default_transcribe(client, img: np.ndarray, model: str = DEFAULT_MODEL,
                        strips: int = 2) -> str:
    return transcribe_page(client, img, model=model, strips=strips,
                           keep_lines=True)


def run(out_dir: str, v2_dir: str, per_doc: int = 12, max_pages: int = 0,
        model: str = DEFAULT_MODEL, strips: int = 2, dpi: int = 300,
        min_sim: float = MIN_SIM, client=None, limit: int = 0,
        ocr_fn=None, transcribe_fn=None, method: str = "montage",
        rows: int = MONTAGE_ROWS, montage_fn=None, timeout_s: float = 90.0,
        retries: int = 2) -> Dict:
    lines_dir = os.path.join(out_dir, "lines")
    pages_dir = os.path.join(out_dir, "pages")
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    report = {"kind": "real_lines", "model": model, "strips": strips,
              "min_sim": min_sim, "dpi": dpi, "method": method, "rows": rows,
              "pages": {}}
    if os.path.exists(report_path):
        report = json.load(open(report_path, encoding="utf-8"))
        print(f"[label] resuming: {len(report['pages'])} pages done")

    plan = pick_pages(v2_dir, per_doc=per_doc, max_pages=max_pages)
    todo = [(p, idx) for p, pages in plan for idx in pages]
    if limit:
        todo = todo[:limit]
    if client is None and not transcribe_fn:
        from anchor_gemini import _client
        client = _client()
        print("[label] gemini client ready")

    labels_path = os.path.join(out_dir, "labels.tsv")
    done = {k for k, v in report["pages"].items() if "lines" in v}
    for k, (src, idx) in enumerate(todo, 1):
        stem = os.path.splitext(os.path.basename(src))[0]
        page_id = f"{stem}_p{idx:03d}"
        if page_id in done:
            continue
        img = None
        for i, page_img, _gt in doc_data.pdf_to_pages(src, dpi=dpi):
            if i == idx:
                img = page_img
                break
            if i > idx:
                break
        if img is None:
            print(f"  [{k}/{len(todo)}] {page_id}: render failed")
            continue
        if method == "montage":
            # English pages exist in the corpus (supreme-court volumes); do not
            # spend Gemini calls on them.
            pre = (ocr_fn or _default_ocr)(img)
            share = (sum(1 for t in pre.tokens if _is_deva(t.text))
                     / max(1, len(pre.tokens)))
            if share < MIN_DEVA_SHARE:
                print(f"  [{k}/{len(todo)}] {page_id}: skipped "
                      f"(devanagari share {share:.0%})", flush=True)
                report["pages"][page_id] = {
                    "source": src, "page": idx, "skipped": "non-devanagari",
                    "lines": []}
                _write(report_path, report)
                continue
        try:
            if method == "montage":
                fn = montage_fn
                if fn is None:
                    fn = lambda *a, **k: transcribe_montage(  # noqa: E731
                        *a, timeout_s=timeout_s, retries=retries, **k)
                row = label_page_montage(client, img, page_id, model,
                                         rows=rows, ocr_fn=ocr_fn,
                                         montage_fn=fn)
            else:
                row = label_page(client, img, page_id, model, strips, min_sim,
                                 ocr_fn=ocr_fn, transcribe_fn=transcribe_fn)
        except Exception as e:  # noqa: BLE001 - one bad page must not kill the run
            print(f"  [{k}/{len(todo)}] {page_id}: FAILED {e}", flush=True)
            report["pages"][page_id] = {"error": str(e)}
            _write(report_path, report)
            continue
        if row.get("gemini_text"):
            with open(os.path.join(pages_dir, f"{page_id}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(row["gemini_text"])
        names = []
        for n, item in enumerate(row["kept"]):
            name = f"{page_id}_l{n:03d}.png"
            doc_data.imwrite_safe(os.path.join(lines_dir, name), item["crop"])
            names.append((name, item["text"], item["sim"]))
        report["pages"][page_id] = {
            "source": src, "page": idx, "ocr_lines": row["ocr_lines"],
            "gemini_lines": row["gemini_lines"], "matched": row["matched"],
            "kept": len(row["kept"]), "dropped": row["dropped"],
            "dropped_reasons": row.get("dropped_reasons"),
            "lines": names,
        }
        _write(report_path, report)
        print(f"  [{k}/{len(todo)}] {page_id}: {row['matched']} matched, "
              f"{len(row['kept'])} kept, {row['dropped']} dropped", flush=True)

    return _finalize(out_dir, report)


def _write(path: str, report: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _finalize(out_dir: str, report: Dict) -> Dict:
    pages = {k: v for k, v in report["pages"].items() if "lines" in v}
    kept = sum(v["kept"] for v in pages.values())
    matched = sum(v["matched"] for v in pages.values())
    digit = 0
    labels_path = os.path.join(out_dir, "labels.tsv")
    with open(labels_path, "w", encoding="utf-8") as f:
        for page_id in sorted(pages):
            for name, text, _s in pages[page_id]["lines"]:
                f.write(f"{name}\t{text}\n")
    for v in pages.values():
        for _name, text, _s in v["lines"]:
            if doc_metrics.digit_tokens(text):
                digit += 1
    summary = {
        "pages": len(pages), "lines_kept": kept, "lines_matched": matched,
        "keep_rate": round(kept / matched, 3) if matched else None,
        "digit_lines": digit,
        "digit_share": round(digit / kept, 3) if kept else None,
        "mean_sim": round(float(np.mean([s for v in pages.values()
                                         for _n, _t, s in v["lines"]])), 4)
        if kept else None,
    }
    report["summary"] = summary
    report["kind"] = "real_lines"
    _write(os.path.join(out_dir, "report.json"), report)
    manifest = {"kind": "real_lines", "name": "deva_real_lines",
                "provenance": f"gemini {report.get('model')} page transcription, "
                              f"DP-aligned to RapidOCR lines (min_sim "
                              f"{report.get('min_sim')})",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary,
                "pages": {k: {kk: vv for kk, vv in v.items() if kk != "lines"}
                          for k, v in pages.items()}}
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("[label] summary:", json.dumps(summary, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser(description="Label real lines with Gemini")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "doc_eval",
                                                  "deva_real_lines"))
    ap.add_argument("--v2-dir", default=os.path.join(BASE_DIR, "data",
                                                     "doc_eval",
                                                     "nepali_pdf_v2"))
    ap.add_argument("--per-doc", type=int, default=12)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strips", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--min-sim", type=float, default=MIN_SIM)
    ap.add_argument("--method", choices=("montage", "page"), default="montage",
                    help="montage: label our line crops directly (1:1); "
                         "page: page transcription + DP alignment")
    ap.add_argument("--rows", type=int, default=MONTAGE_ROWS,
                    help="line crops per montage call")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-call Gemini timeout (seconds)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--json", default=None, help="copy of report.json")
    args = ap.parse_args()
    report = run(args.out, args.v2_dir, per_doc=args.per_doc,
                 max_pages=args.max_pages, model=args.model, strips=args.strips,
                 dpi=args.dpi, min_sim=args.min_sim, limit=args.limit,
                 method=args.method, rows=args.rows, timeout_s=args.timeout,
                 retries=args.retries)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": report["summary"]}, f, indent=2,
                      ensure_ascii=False)


if __name__ == "__main__":
    main()
