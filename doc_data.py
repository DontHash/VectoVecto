"""
doc_data.py — evaluation data builder for the document-restore product.

Zero human transcription strategy:
  1. Synthetic invoices rendered by us -> exact UTF-8 ground truth.
  2. Any openly-licensed PDF rendered at high DPI -> ground truth comes from the
     PDF's own text layer (pypdfium2). The clean-render -> text-layer CER is the
     extraction noise floor and is recorded per page.
  3. Real photos from public HF datasets with upstream text annotation (SROIE,
     CORD) -> no clean reference; internal eval only, never redistributed.
  4. A demo PDF built with reportlab exercises the PDF path fully offline (tests).

Dataset layout (written under --out, gitignored):
    <out>/<name>/pages/<id>_clean.png
    <out>/<name>/pages/<id>_degraded.png
    <out>/<name>/gt/<id>.txt
    <out>/<name>/manifest.json

CLI:
    python doc_data.py --out data/doc_eval --synthetic 20 --levels mild,medium
    python doc_data.py --out data/doc_eval --pdf openbook.pdf --dpi 250
    python doc_data.py --out data/doc_eval --hf sroie cord --hf-pages 30
    python doc_data.py --demo-pdf out/demo_invoice.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from degradation_document import degrade_page  # noqa: E402


# ---------------------------------------------------------------------------
# text normalization (shared with the eval harness)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """CRLF->LF, collapse spaces/tabs, strip. Keep case and punctuation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = []
    try:
        import matplotlib
        mpl_fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        candidates += [
            os.path.join(mpl_fonts, "DejaVuSans.ttf"),
            os.path.join(mpl_fonts, "DejaVuSerif.ttf"),
            os.path.join(mpl_fonts, "DejaVuSansMono.ttf"),
        ]
    except Exception:
        pass
    candidates += [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# synthetic invoices
# ---------------------------------------------------------------------------

_VENDORS = ["ACME Supplies Ltd", "Northwind Traders", "Globex Corporation",
            "Initech Services", "Umbrella Logistics"]
_ITEMS = [("Widget A-100", 3, "120.00"), ("Gadget B-220", 1, "899.50"),
          ("Cable C-05", 10, "12.75"), ("Service Fee", 1, "250.00"),
          ("License D-9", 2, "399.99"), ("Support Plan", 1, "149.00")]


def render_synthetic_invoice(seed: int = 0, dpi: int = 300) -> Tuple[np.ndarray, str]:
    """Returns (BGR uint8 page, ground-truth text in reading order).

    Rendered at 300 DPI by default so degradations act like a real phone photo
    (text stays ~40-90 px tall before downscale), not like shrinking an
    already-low-res thumbnail.
    """
    rng = np.random.default_rng(seed)
    s = dpi / 150.0
    w, h = int(1240 * s), int(1754 * s)

    def sc(*vals):
        return tuple(int(round(v * s)) for v in vals)

    page = Image.new("RGB", (w, h), (250, 249, 246))
    d = ImageDraw.Draw(page)
    f_title = _font(int(46 * s))
    f_head = _font(int(26 * s))
    f_body = _font(int(24 * s))
    f_mono = _font(int(24 * s))

    lines: List[str] = []
    inv_no = int(rng.integers(10000, 99999))
    vendor = _VENDORS[int(rng.integers(0, len(_VENDORS)))]
    date = f"2026-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}"
    title = "INVOICE"
    d.text(sc(80, 70), title, font=f_title, fill=(15, 15, 15))
    lines.append(title)

    d.text(sc(80, 150), f"Vendor: {vendor}", font=f_head, fill=(30, 30, 30))
    d.text(sc(80, 185), f"Invoice No: {inv_no}", font=f_body, fill=(30, 30, 30))
    d.text(sc(80, 218), f"Date: {date}", font=f_body, fill=(30, 30, 30))
    lines += [f"Vendor: {vendor}", f"Invoice No: {inv_no}", f"Date: {date}"]

    d.text(sc(80, 300), "Description", font=f_head, fill=(0, 0, 0))
    d.text(sc(620, 300), "Qty", font=f_head, fill=(0, 0, 0))
    d.text(sc(760, 300), "Unit", font=f_head, fill=(0, 0, 0))
    d.text(sc(980, 300), "Amount", font=f_head, fill=(0, 0, 0))
    lines.append("Description Qty Unit Amount")
    d.line(sc(80, 340, 1160, 340), fill=(120, 120, 120), width=max(1, int(2 * s)))

    total = 0.0
    y = 360
    n_items = int(rng.integers(2, 5))
    picked = [int(i) for i in rng.choice(len(_ITEMS), size=n_items, replace=False)]
    for idx in picked:
        name, qty, unit = _ITEMS[idx]
        qty = int(qty) + int(rng.integers(0, 3))
        unit_f = float(unit)
        amount = qty * unit_f
        total += amount
        row = f"{name} {qty} {unit_f:.2f} {amount:.2f}"
        d.text(sc(80, y), name, font=f_body, fill=(25, 25, 25))
        d.text(sc(620, y), str(qty), font=f_body, fill=(25, 25, 25))
        d.text(sc(760, y), f"{unit_f:.2f}", font=f_mono, fill=(25, 25, 25))
        d.text(sc(980, y), f"{amount:.2f}", font=f_mono, fill=(25, 25, 25))
        lines.append(row)
        y += 44

    tax = round(total * 0.13, 2)
    grand = round(total + tax, 2)
    d.line(sc(700, y + 6, 1160, y + 6), fill=(120, 120, 120), width=max(1, int(2 * s)))
    d.text(sc(760, y + 20), "Subtotal", font=f_head, fill=(0, 0, 0))
    d.text(sc(980, y + 20), f"{total:.2f}", font=f_mono, fill=(0, 0, 0))
    d.text(sc(760, y + 56), "VAT 13%", font=f_head, fill=(0, 0, 0))
    d.text(sc(980, y + 56), f"{tax:.2f}", font=f_mono, fill=(0, 0, 0))
    d.text(sc(760, y + 100), "TOTAL", font=f_title, fill=(10, 10, 10))
    d.text(sc(980, y + 104), f"{grand:.2f}", font=f_title, fill=(10, 10, 10))
    lines += [f"Subtotal {total:.2f}", f"VAT 13% {tax:.2f}", f"TOTAL {grand:.2f}"]

    d.text(sc(80, 1754 - 90), "Thank you for your business.", font=f_body, fill=(60, 60, 60))
    lines.append("Thank you for your business.")

    arr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    return arr, normalize_text("\n".join(lines))


# ---------------------------------------------------------------------------
# PDF -> pages with text-layer GT
# ---------------------------------------------------------------------------

def pdf_to_pages(pdf_path: str, dpi: int = 200):
    """Yield (page_index, BGR image, text-layer GT, clean-render CER floor).

    NOTE: text-layer order is the PDF's internal order; multi-column layouts may
    not be in reading order. The per-page floor CER (OCR of the *clean* render
    vs its own text layer) quantifies that noise for every method comparison.
    """
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72.0
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil().convert("RGB")
        text = normalize_text(page.get_textpage().get_text_range())
        arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        yield i, arr, text


def build_pdf_dataset(pdf_paths: List[str], out_dir: str, name: str | None = None,
                      dpi: int = 200, levels=("medium",), max_pages: int = 0,
                      seed: int = 1234) -> Dict:
    """Render PDFs, degrade pages, write dataset + manifest. Returns manifest."""
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    count = 0
    for pdf_path in pdf_paths:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        for idx, img, gt in pdf_to_pages(pdf_path, dpi=dpi):
            if not gt:
                continue
            if max_pages and count >= max_pages:
                break
            level = levels[count % len(levels)]
            page_seed = seed + count
            pid = f"{stem}_p{idx:03d}"
            clean_path = os.path.join(pages_dir, f"{pid}_clean.png")
            deg_path = os.path.join(pages_dir, f"{pid}_degraded.png")
            gt_path = os.path.join(gt_dir, f"{pid}.txt")
            cv2.imwrite(clean_path, img)
            cv2.imwrite(deg_path, degrade_page(img, seed=page_seed, level=level))
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(gt)
            entries.append({
                "id": pid, "source": pdf_path, "page": idx, "level": level,
                "seed": page_seed, "gt_chars": len(gt),
                "clean": os.path.relpath(clean_path, out_dir),
                "degraded": os.path.relpath(deg_path, out_dir),
                "gt": os.path.relpath(gt_path, out_dir),
            })
            count += 1
        if max_pages and count >= max_pages:
            break
    manifest = {"kind": "pdf", "name": name or "pdf_eval", "dpi": dpi,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# synthetic dataset
# ---------------------------------------------------------------------------

def build_synthetic_dataset(out_dir: str, n: int = 20,
                            levels=("mild", "medium"), seed: int = 100) -> Dict:
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    for i in range(n):
        img, gt = render_synthetic_invoice(seed)
        level = levels[i % len(levels)]
        page_seed = seed * 1000 + i
        pid = f"inv_{i:04d}"
        clean_path = os.path.join(pages_dir, f"{pid}_clean.png")
        deg_path = os.path.join(pages_dir, f"{pid}_degraded.png")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        cv2.imwrite(clean_path, img)
        cv2.imwrite(deg_path, degrade_page(img, seed=page_seed, level=level))
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt)
        entries.append({
            "id": pid, "source": "synthetic", "level": level, "seed": page_seed,
            "gt_chars": len(gt),
            "clean": os.path.relpath(clean_path, out_dir),
            "degraded": os.path.relpath(deg_path, out_dir),
            "gt": os.path.relpath(gt_path, out_dir),
        })
        seed += 1
    manifest = {"kind": "synthetic", "name": "synthetic_invoices",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# real-photo datasets from Hugging Face (annotations, no clean reference)
# ---------------------------------------------------------------------------

# Verified mirrors (2026-09): parquet with embedded PIL images + text.
HF_SOURCES = {
    "sroie": "jsdnrs/ICDAR2019-SROIE",   # ICDAR'19 scanned receipts, line words
    "cord": "naver-clova-ix/cord-v2",    # receipts, structured JSON GT (CC-BY-4.0)
}


def _collect_text_leaves(node, out: List[str]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _collect_text_leaves(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_text_leaves(v, out)
    elif isinstance(node, str) and node.strip():
        out.append(node.strip())


def gt_from_sroie(example: Dict) -> str:
    """Line-level `words` annotation -> newline text."""
    words = example.get("words") or []
    return normalize_text("\n".join(str(w) for w in words))


def gt_from_cord(example: Dict) -> str:
    """CORD `ground_truth` JSON (gt_parse tree) -> text leaves in JSON order."""
    raw = example.get("ground_truth")
    if not raw:
        return ""
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return normalize_text(raw)
    gt = data.get("gt_parse", data) if isinstance(data, dict) else data
    leaves: List[str] = []
    _collect_text_leaves(gt, leaves)
    return normalize_text("\n".join(leaves))


def gt_from_generic(example: Dict) -> str:
    """Any HF repo: words / text / label / ground_truth fields."""
    if example.get("words"):
        return gt_from_sroie(example)
    for key in ("text", "label", "ground_truth"):
        v = example.get(key)
        if isinstance(v, str) and v.strip():
            return gt_from_cord(example) if key == "ground_truth" else normalize_text(v)
    return ""


def _example_image(example: Dict):
    """First PIL image field; falls back to readable image_path strings."""
    from PIL import Image as PILImage
    for v in example.values():
        if hasattr(v, "mode") and hasattr(v, "size"):
            return v if v.mode == "RGB" else v.convert("RGB")
    for key in ("image", "image_path"):
        p = example.get(key)
        if isinstance(p, str) and os.path.exists(p):
            return PILImage.open(p).convert("RGB")
    return None


def build_hf_dataset(source: str, out_dir: str, max_pages: int = 30,
                     split: str | None = None, offset: int = 0) -> Dict:
    """Stream real photos + text GT from Hugging Face into the eval layout.

    Real captures are *not* degraded: clean == degraded == the original photo,
    so 'clean' is not an extraction ceiling here — judge raw vs restore and use
    cer_bag / bag digit CER (annotation order is not guaranteed to match the
    visual reading order). Internal evaluation only; never redistribute.
    """
    from datasets import load_dataset

    repo = HF_SOURCES.get(source, source)
    gt_fn, default_split = {
        "sroie": (gt_from_sroie, "train"),
        "cord": (gt_from_cord, "train"),
    }.get(source, (gt_from_generic, "train"))
    slug = source.replace("/", "__")
    ds = load_dataset(repo, split=split or default_split, streaming=True)

    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    for idx, example in enumerate(ds):
        if idx < offset:
            continue
        if max_pages and len(entries) >= max_pages:
            break
        gt = gt_fn(example)
        img = _example_image(example)
        if img is None or len(gt) < 20:
            continue
        pid = f"{slug}_{idx:05d}"
        page_path = os.path.join(pages_dir, f"{pid}.png")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(page_path, arr)
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt)
        entries.append({
            "id": pid, "source": repo, "page": idx, "level": "real", "real": True,
            "seed": None, "gt_chars": len(gt),
            "clean": os.path.relpath(page_path, out_dir),
            "degraded": os.path.relpath(page_path, out_dir),
            "gt": os.path.relpath(gt_path, out_dir),
        })
    manifest = {
        "kind": "real", "name": f"real_{slug}", "source": repo,
        "split": split or default_split, "offset": offset,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# demo PDF (offline PDF-path exercise)
# ---------------------------------------------------------------------------

def make_demo_pdf(path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, 780, "INVOICE 1200.00")
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "Vendor: ACME Supplies Ltd")
    c.drawString(72, 730, "Date: 2026-09-20")
    c.drawString(72, 700, "Widget A-100  3  120.00  360.00")
    c.drawString(72, 680, "Gadget B-220  1  899.50  899.50")
    c.drawString(72, 650, "TOTAL  1424.14")
    c.showPage()
    c.save()


def load_dataset(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for e in manifest["entries"]:
        e["_clean_path"] = os.path.join(data_dir, e["clean"])
        e["_degraded_path"] = os.path.join(data_dir, e["degraded"])
        e["_gt_path"] = os.path.join(data_dir, e["gt"])
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build document eval datasets")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "doc_eval"))
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--levels", default="mild,medium")
    ap.add_argument("--pdf", nargs="*", default=[])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--demo-pdf", default=None)
    ap.add_argument("--hf", nargs="*", default=[],
                    help="HF sources for real photos: sroie, cord, or a repo id")
    ap.add_argument("--hf-pages", type=int, default=30)
    ap.add_argument("--hf-split", default=None)
    ap.add_argument("--hf-offset", type=int, default=0)
    args = ap.parse_args()

    if args.demo_pdf:
        make_demo_pdf(args.demo_pdf)
        print(f"wrote {args.demo_pdf}")
        return

    levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())
    if args.synthetic:
        m = build_synthetic_dataset(os.path.join(args.out, "synthetic"),
                                    n=args.synthetic, levels=levels, seed=args.seed)
        print(f"synthetic: {len(m['entries'])} pages -> {os.path.join(args.out, 'synthetic')}")
    if args.pdf:
        m = build_pdf_dataset(args.pdf, os.path.join(args.out, "pdf"), dpi=args.dpi,
                              levels=levels, max_pages=args.max_pages, seed=args.seed)
        print(f"pdf: {len(m['entries'])} pages -> {os.path.join(args.out, 'pdf')}")
    for source in args.hf:
        slug = source.replace("/", "__")
        out = os.path.join(args.out, f"real_{slug}")
        m = build_hf_dataset(source, out, max_pages=args.hf_pages,
                             split=args.hf_split, offset=args.hf_offset)
        print(f"real[{source}]: {len(m['entries'])} pages -> {out}")


if __name__ == "__main__":
    main()
