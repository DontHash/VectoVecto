"""
build_digit_crop_sheet.py — digit-crop review sheet for human correction (N5).

Renders *unlabeled* modern pages (the unused `nepali_pdf_sources_v2` PDFs and
unfrozen Cornell textbook pages), cuts **digit-dominant** engine token crops,
and writes a TSV + HTML viewer (+ optional montage sheets) so the wrong
readings can be corrected quickly. `--import` validates the corrections and
stores them as a recognizer training set (`lines/` + `labels.tsv`, the format
`scripts/export_training_data.py --real-dir` consumes).

Never touches frozen evaluation pages (excluded via the in-dir manifests).

Workflow:
    python scripts/build_digit_crop_sheet.py              # build the sheet
    # edit out/labeling/digit_crops/review.tsv  (last column)
    python scripts/build_digit_crop_sheet.py --import     # validate + store

The corrected crops are training ground truth only; they are never merged into
a frozen evaluation set (see docs/EVALUATION.md).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import json
import os
import sys
import time
import unicodedata
import zlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import doc_data  # noqa: E402
from degradation_document import degrade_page  # noqa: E402
from document_ocr import _crop_with_pad, get_backend  # noqa: E402

SRC_V2 = os.path.join(BASE_DIR, "data", "doc_eval", "nepali_pdf_sources_v2")
SRC_CORNELL = os.path.join(BASE_DIR, "data", "doc_eval",
                           "nepali_textbook_scans")
FROZEN_V2 = os.path.join(BASE_DIR, "data", "doc_eval", "nepali_pdf_v2",
                         "manifest.json")
FROZEN_CORNELL = os.path.join(BASE_DIR, "data", "doc_eval", "cornell_real",
                              "manifest.json")
OUT_DEFAULT = os.path.join(BASE_DIR, "out", "labeling", "digit_crops")
DEST_DEFAULT = os.path.join(BASE_DIR, "data", "doc_eval", "digit_lines_v1")

# Characters a digit token may reasonably contain, on top of all Devanagari
# code points and ASCII/Latin letters (a majority-digit token can still carry a
# stray consonant). Anything else is reported, not dropped: the human decides.
_COMMON_PUNCT = set(" \t.,;:/-()[]{}%+=*'\"&|<>@#_!?~^")


def sanitize_field(text: str) -> str:
    """Collapse tabs/newlines so one token cannot corrupt a TSV row."""
    return " ".join((text or "").replace("\t", " ").replace("\r", " ")
                    .replace("\n", " ").split())


def normalize_text(text: str) -> str:
    """NFC + whitespace-collapsed label text (script is preserved as printed)."""
    return sanitize_field(unicodedata.normalize("NFC", text or ""))


def frozen_pages(manifest_path: str):
    """{(source basename, page)} used by a frozen set."""
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    return {(os.path.basename(e.get("source", "")), int(e.get("page", -1)))
            for e in m["entries"] if e.get("source")}


def digit_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isdigit()) / len(chars)


def is_digit_crop(text: str, min_ratio: float) -> bool:
    return any(c.isdigit() for c in text) and digit_ratio(text) >= min_ratio


def collect_crops(pages, be, out_dir, max_per_page, crop_rows, min_ratio,
                  degrade_level="none", seed=42, prefix=""):
    for pid, img in pages:
        if degrade_level != "none":
            page_seed = (seed + zlib.crc32(pid.encode("utf-8"))) & 0x7FFFFFFF
            img = degrade_page(img, seed=page_seed, level=degrade_level)
        res = be.run(img, lang="ne")
        taken = 0
        for i, t in enumerate(res.tokens):
            if taken >= max_per_page:
                break
            text = sanitize_field(t.text)
            if not is_digit_crop(text, min_ratio):
                continue
            crop = _crop_with_pad(img, t.bbox, pad_ratio=0.02)
            if crop.size == 0 or min(crop.shape[:2]) < 6:
                continue
            big = cv2.resize(crop, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_LANCZOS4)
            cid = f"{prefix}{pid}_c{i:02d}"
            doc_data.imwrite_safe(os.path.join(out_dir, "crops", f"{cid}.png"),
                                  big)
            crop_rows.append({"crop_id": cid, "page_id": pid,
                              "engine_guess": text, "true_text": text})
            taken += 1
        print(f"  {pid}: {taken} crops")


def render_pdf_pages(pdf: str, dpi: int, max_pages: int, skip: set,
                     start: int = 1):
    base = os.path.basename(pdf)
    for idx, img, _tl in doc_data.pdf_to_pages(pdf, dpi=dpi):
        if idx < start or (base, idx) in skip:
            continue
        if max_pages <= 0:
            break
        ink = float((img.mean(axis=2) < 128).mean())
        if ink < 0.01:
            continue
        max_pages -= 1
        yield f"{os.path.splitext(base)[0][:20]}_p{idx:03d}", img


def write_sheet(out_dir: str, rows, min_ratio: float,
                degrade: str = "none") -> None:
    with open(os.path.join(out_dir, "review.tsv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("crop_id\tpage_id\tengine_guess\ttrue_text\n")
        for r in rows:
            f.write(f"{r['crop_id']}\t{r['page_id']}\t"
                    f"{r['engine_guess']}\t{r['true_text']}\n")
    parts = ["""<!doctype html><meta charset="utf-8">
<title>Digit-crop review</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 24px; }
 img { height: 56px; border: 1px solid #bbb; background: #fff;
       image-rendering: auto; }
 td { padding: 6px 10px; border-bottom: 1px solid #eee; }
 code { background: #f4f4f4; padding: 1px 4px; }
</style>
<h1>Digit-crop review</h1>
<p>Edit the last column of <code>review.tsv</code> (next to this file) so
every <code>true_text</code> matches its crop. Leave a row alone if the
engine guess is right; blank a row to reject it. Then run
<code>--import</code>.</p>
<table><tr><th>crop</th><th>page</th><th>engine guess</th></tr>
"""]
    for r in rows:
        parts.append(
            f"<tr><td><img src='crops/{html.escape(r['crop_id'])}.png'></td>"
            f"<td>{html.escape(r['page_id'])}</td>"
            f"<td>{html.escape(r['engine_guess'])}</td></tr>\n")
    parts.append("</table>\n")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("".join(parts))
    with open(os.path.join(out_dir, "LABELING.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(f"""# Digit-crop review ({len(rows)} crops, digit ratio >= {min_ratio:g}, degrade={degrade})

1. Open `index.html` (crop images + the engine's guess).
2. Edit `review.tsv`: fix the **true_text** column where the guess is wrong.
   Keep Devanagari digits as printed (Latin digits only where the page prints
   Latin). Blank a `true_text` to reject a bad crop. Leave correct rows alone.
3. Run `python scripts/build_digit_crop_sheet.py --import` to validate and
   store the corrections as a training set (`lines/` + `labels.tsv`).

These crops are **training ground truth**, never a frozen evaluation set.
""")


def write_montage(out_dir: str, rows, cols: int = 2, per_sheet: int = 15,
                  cell_h: int = 64, cell_w: int = 820) -> None:
    """Contact sheets (image + numeric index) for fast scanning/review."""
    index_path = os.path.join(out_dir, "montage_index.tsv")
    with open(index_path, "w", encoding="utf-8", newline="\n") as fi:
        fi.write("idx\tcrop_id\tengine_guess\n")
        for i, r in enumerate(rows):
            fi.write(f"{i:03d}\t{r['crop_id']}\t{r['engine_guess']}\n")
    sheet = 0
    for start in range(0, len(rows), cols * per_sheet):
        block = rows[start:start + cols * per_sheet]
        grid = np.full((per_sheet * cell_h, cols * cell_w, 3), 255, np.uint8)
        for j, r in enumerate(block):
            img = doc_data.imread_safe(
                os.path.join(out_dir, "crops", f"{r['crop_id']}.png"))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min((cell_h - 16) / max(1, h), (cell_w - 12) / max(1, w))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            x = (j % cols) * cell_w + 6
            y = (j // cols) * cell_h + 6
            grid[y:y + nh, x:x + nw] = resized
            cv2.putText(grid, f"{start + j:03d}", (x, y + nh + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1,
                        cv2.LINE_AA)
        doc_data.imwrite_safe(os.path.join(out_dir, f"montage_{sheet:02d}.png"),
                              grid)
        sheet += 1
    print(f"[montage] {sheet} sheet(s), index -> {index_path}")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def do_import(out_dir: str, dest: str, min_ratio: float, force: bool) -> None:
    tsv = os.path.join(out_dir, "review.tsv")
    if not os.path.exists(tsv):
        raise SystemExit(f"no review.tsv at {tsv}; build the sheet first")
    labels_path = os.path.join(dest, "labels.tsv")
    if os.path.exists(labels_path) and not force:
        raise SystemExit(f"{labels_path} exists; pass --force to overwrite")
    with open(tsv, encoding="utf-8", newline="") as f:
        # OCR text can start with a literal quote; never let csv treat it as
        # a quoting character (it would swallow the rest of the row).
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    lines_dir = os.path.join(dest, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    labels, rejected = [], []
    for r in rows:
        cid = (r.get("crop_id") or "").strip()
        if not cid:
            continue
        src = os.path.join(out_dir, "crops", f"{cid}.png")
        if not os.path.exists(src):
            rejected.append({"crop_id": cid, "reason": "missing_crop"})
            continue
        raw = r.get("true_text")
        if raw is None:
            rejected.append({"crop_id": cid, "reason": "malformed_tsv_row"})
            continue
        text = normalize_text(raw)
        if not text:
            rejected.append({"crop_id": cid, "reason": "rejected_blank"})
            continue
        if not any(c.isdigit() for c in text):
            rejected.append({"crop_id": cid, "reason": "no_digit"})
            continue
        img = doc_data.imread_safe(src)
        if img is None or img.size == 0:
            rejected.append({"crop_id": cid, "reason": "unreadable_crop"})
            continue
        name = f"{cid}.png"
        if not doc_data.imwrite_safe(os.path.join(lines_dir, name), img):
            rejected.append({"crop_id": cid, "reason": "copy_failed"})
            continue
        labels.append((name, text))

    with open(labels_path, "w", encoding="utf-8", newline="\n") as f:
        for name, text in labels:
            f.write(f"{name}\t{text}\n")

    manifest = {
        "name": "digit_lines_v1",
        "purpose": "N5 real digit ground truth for a digit recognizer",
        "source_review_tsv": os.path.relpath(tsv, BASE_DIR).replace("\\", "/"),
        "source_note": ("unfrozen modern pages: unused nepali_pdf_sources_v2 "
                        "PDFs + unfrozen Cornell textbook scans; frozen pages "
                        "excluded at collection time"),
        "min_digit_ratio": min_ratio,
        "n_labels": len(labels),
        "n_rejected": len(rejected),
        "labels_sha256": _sha256(labels_path),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rejected": rejected,
    }
    with open(os.path.join(dest, "import_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dest, "PROVENANCE.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(f"""# digit_lines_v1

Real **digit** line crops for recognizer training (N5), produced from the
human-reviewed sheet `{manifest['source_review_tsv']}`.

- {len(labels)} labels (rejected {len(rejected)}), created {manifest['created']}.
- Source pages are unfrozen modern pages (born-digital PDFs + Cornell textbook
  scans); frozen evaluation pages were excluded when the sheet was built.
- This is **training ground truth**. It is not a frozen evaluation set and
  must never be merged into one.
- labels.tsv sha256 `{manifest['labels_sha256']}`.

Consume with:
`python scripts/export_training_data.py --real-dir {os.path.relpath(dest, BASE_DIR).replace(chr(92), '/')} ...`
""")
    print(f"[import] {len(labels)} labels, {len(rejected)} rejected -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Digit-crop review sheet (N5)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max-crops", type=int, default=250)
    ap.add_argument("--max-per-page", type=int, default=12)
    ap.add_argument("--pages-per-pdf", type=int, default=4)
    ap.add_argument("--cornell-pages", type=int, default=10,
                    help="content pages per Cornell book (from page 6)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--min-digit-ratio", type=float, default=0.7,
                    help="keep only tokens at least this fraction digits "
                         "(0 = any token containing a digit, the old behaviour)")
    ap.add_argument("--montage", action="store_true",
                    help="also write contact-sheet PNGs for fast review")
    ap.add_argument("--degrade", choices=("none", "mild", "medium", "heavy"),
                    default="none",
                    help="degrade each rendered page before OCR (targets the "
                         "photo/scan domain where digit reading fails)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="validate corrections and store the training set")
    ap.add_argument("--dest", default=DEST_DEFAULT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.do_import:
        do_import(args.out, args.dest, args.min_digit_ratio, args.force)
        return

    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    be = get_backend("rapidocr")
    rows = []
    prefix = "" if args.degrade == "none" else f"{args.degrade[:3]}_"

    skip_v2 = frozen_pages(FROZEN_V2)
    used_docs = {s for s, _ in skip_v2}
    pdfs = [p for p in sorted(glob.glob(os.path.join(SRC_V2, "*.pdf")))
            if os.path.basename(p) not in used_docs]
    print(f"[sheet] unused v2 sources: {len(pdfs)} (degrade={args.degrade})")
    for pdf in pdfs:
        if len(rows) >= args.max_crops:
            break
        pages = list(render_pdf_pages(pdf, args.dpi, args.pages_per_pdf,
                                      skip_v2))
        print(f"  [{os.path.basename(pdf)[:28]}] {len(pages)} pages")
        collect_crops(pages, be, args.out, args.max_per_page, rows,
                      args.min_digit_ratio, args.degrade, args.seed,
                      prefix=prefix)

    skip_c = frozen_pages(FROZEN_CORNELL)
    cornell = sorted(glob.glob(os.path.join(SRC_CORNELL, "*.pdf")))
    for pdf in cornell:
        if len(rows) >= args.max_crops:
            break
        pages = list(render_pdf_pages(pdf, args.dpi, args.cornell_pages,
                                      skip_c, start=6))
        print(f"  [{os.path.basename(pdf)[:28]}] {len(pages)} pages")
        collect_crops(pages, be, args.out, args.max_per_page, rows,
                      args.min_digit_ratio, args.degrade, args.seed,
                      prefix=prefix)

    rows = rows[:args.max_crops]
    write_sheet(args.out, rows, args.min_digit_ratio, args.degrade)
    if args.montage:
        write_montage(args.out, rows)
    print(f"[sheet] {len(rows)} digit crops -> {args.out}")


if __name__ == "__main__":
    main()
