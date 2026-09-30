"""
build_cornell_labeling.py — real-scan ground-truth slice (Track D).

Renders content pages from the Cornell eCommons Nepali textbook scans
(collection 1813/24179, no text layer), pre-fills each page with the shipped
pipeline reading, and writes a side-by-side labeling sheet.

Workflow:
    python scripts/build_cornell_labeling.py
    # correct data/doc_eval/cornell_real/gt/<id>.txt against the page images
    python evals/harness/eval_freeze.py --name cornell_real_v1 \\
        --out evals/manifests/cornell_real_v1.json \\
        --data-dir data/doc_eval/cornell_real --license "Cornell eCommons 1813 (scans)"

The pre-filled text is a starting point, not GT: only the corrected files are.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.core import data as doc_data  # noqa: E402
from veriscript.document.pipeline import run_document_pipeline  # noqa: E402

DEFAULT_PDF_DIR = os.path.join(BASE_DIR, "data", "doc_eval",
                               "nepali_textbook_scans")
DEFAULT_OUT = os.path.join(BASE_DIR, "data", "doc_eval", "cornell_real")
DEFAULT_SHEET = os.path.join(BASE_DIR, "out", "labeling", "cornell")


def prefill(img, lang: str) -> str:
    """Shipped pipeline reading as the labeling starting point."""
    res = run_document_pipeline(img, backend="rapidocr", lang=lang)
    return res.ocr.text.strip() + "\n"


def build(pdf_dir: str, out_dir: str, sheet_dir: str, pages_per_book: int,
          start_page: int, dpi: int, lang: str, limit_books: int = 0,
          min_ink: float = 0.02) -> dict:
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    if limit_books:
        pdfs = pdfs[:limit_books]
    entries = []
    for pdf in pdfs:
        stem = os.path.splitext(pdf)[0]
        taken = 0
        for idx, img, _tl in doc_data.pdf_to_pages(
                os.path.join(pdf_dir, pdf), dpi=dpi):
            if idx < start_page:
                continue
            if taken >= pages_per_book:
                break
            ink = float((img.mean(axis=2) < 128).mean())
            if ink < min_ink:  # blank verso / divider: nothing to label
                print(f"  [skip] {stem}_p{idx:03d} blank (ink {ink:.4f})")
                continue
            pid = f"{stem}_p{idx:03d}"
            png_rel = f"pages/{pid}.png"
            gt_rel = f"gt/{pid}.txt"
            doc_data.imwrite_safe(os.path.join(out_dir, png_rel), img)
            text = prefill(img, lang)
            with open(os.path.join(out_dir, gt_rel), "w", encoding="utf-8",
                      newline="\n") as f:
                f.write(text)
            entries.append({
                "id": pid, "source": pdf, "page": idx, "real": True,
                "clean": png_rel, "degraded": png_rel, "gt": gt_rel,
                "prefill_chars": len(text),
            })
            print(f"  [{len(entries)}] {pid} ({text.count(chr(10))} lines)")
            taken += 1
        if taken == 0:
            print(f"  [warn] no pages taken from {pdf}")

    manifest = {
        "kind": "real_pages", "name": "cornell_real", "dpi": dpi,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Cornell eCommons collection 1813/24179 (scans, no text layer)",
        "prefill": "document_pipeline rapidocr (NOT ground truth until corrected)",
        "entries": entries,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_sheet(sheet_dir, out_dir, entries)
    return manifest


def write_sheet(sheet_dir: str, data_dir: str, entries) -> None:
    os.makedirs(sheet_dir, exist_ok=True)
    parts = ["""<!doctype html><meta charset="utf-8">
<title>Cornell real-scan labeling</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 24px; }
 .page { border-top: 2px solid #ccc; padding: 16px 0; }
 .row { display: flex; gap: 24px; align-items: flex-start; }
 img { max-width: 46vw; border: 1px solid #999; }
 pre { white-space: pre-wrap; max-width: 46vw; background: #f7f7f7;
       padding: 12px; border: 1px solid #ddd; }
 code { background: #eee; padding: 1px 4px; }
</style>
<h1>Cornell real-scan labeling</h1>
<p>Correct <code>gt/&lt;id&gt;.txt</code> in
<code>data/doc_eval/cornell_real/</code> so it matches the page exactly
(reading order, printed digits as written). The text below is the engine
pre-fill, not ground truth. Keep one line per printed line.</p>
"""]
    for e in entries:
        png = os.path.relpath(os.path.join(data_dir, e["clean"]),
                              sheet_dir).replace("\\", "/")
        with open(os.path.join(data_dir, e["gt"]), encoding="utf-8") as f:
            text = f.read()
        parts.append(f"""<div class="page">
<h2>{html.escape(e['id'])}</h2>
<div class="row">
  <img src="{png}" alt="{html.escape(e['id'])}">
  <pre>{html.escape(text)}</pre>
</div>
<p>edit: <code>{html.escape(e['gt'])}</code></p>
</div>
""")
    with open(os.path.join(sheet_dir, "index.html"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("".join(parts))
    with open(os.path.join(sheet_dir, "LABELING.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(f"""# Cornell real-scan labeling

1. Open `index.html` (image left, engine pre-fill right).
2. Edit `data/doc_eval/cornell_real/gt/<id>.txt` to match the page exactly:
   reading order, one line per printed line, Devanagari digits as printed.
   The pre-fill is the engine reading, **not** ground truth.
3. When done, tell the agent to freeze and measure:
   `python evals/harness/eval_freeze.py --name cornell_real_v1 --out
   evals/manifests/cornell_real_v1.json --data-dir data/doc_eval/cornell_real`

Pages: {len(entries)}
""")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Cornell labeling slice")
    ap.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--pages-per-book", type=int, default=4)
    ap.add_argument("--start-page", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit-books", type=int, default=0)
    ap.add_argument("--min-ink", type=float, default=0.02,
                    help="skip pages whose ink fraction is below this (blank "
                         "versos and dividers have nothing to label)")
    args = ap.parse_args()
    m = build(args.pdf_dir, args.out, args.sheet, args.pages_per_book,
              args.start_page, args.dpi, args.lang, args.limit_books,
              args.min_ink)
    print(f"[labeling] {len(m['entries'])} pages -> {args.out}")
    print(f"[labeling] sheet -> {os.path.join(args.sheet, 'index.html')}")


if __name__ == "__main__":
    main()
