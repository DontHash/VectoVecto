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
import glob
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


# ---------------------------------------------------------------------------
# Unicode-safe image IO
#
# OpenCV's imread/imwrite pass the Python path down as a UTF-8 byte string that
# Windows then interprets in the ANSI codepage: any non-ASCII filename lands on
# disk mangled ("\u0906..." -> cp1252 mojibake) and only OpenCV can read it
# back — os.path.exists / freeze hashing / any Python tooling all see a
# different name. The eval harness must not build on that. imencode+tofile and
# fromfile+imdecode go through Python's path handling and stay consistent.
# ---------------------------------------------------------------------------

def imwrite_safe(path: str, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def imread_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

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
                      max_pages_per_pdf: int = 0, seed: int = 1234,
                      degrade: bool = True, gt_validity: bool = False) -> Dict:
    """Render PDFs, degrade pages, write dataset + manifest. Returns manifest.

    `degrade=False` marks the set as real (born-digital): clean == degraded ==
    the render, and metrics are judged against the PDF's own text layer (whose
    order is internal, hence bagCER). Use for real Nepali government PDFs.

    `gt_validity=True` audits every page's text layer with
    `doc_metrics.devanagari_validity` and stores `gt_invalid_tokens` /
    `gt_invalid_token_rate` on the entry: modern Nepali PDFs routinely have
    corrupt ToUnicode maps, so the GT quality must travel with the data.
    """
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    count = 0
    for pdf_path in pdf_paths:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        per_pdf = 0
        for idx, img, gt in pdf_to_pages(pdf_path, dpi=dpi):
            if not gt:
                continue
            if max_pages and count >= max_pages:
                break
            if max_pages_per_pdf and per_pdf >= max_pages_per_pdf:
                break
            per_pdf += 1
            level = levels[count % len(levels)] if degrade else "real"
            page_seed = seed + count
            pid = f"{stem}_p{idx:03d}"
            clean_path = os.path.join(pages_dir, f"{pid}_clean.png")
            deg_path = os.path.join(pages_dir, f"{pid}_degraded.png")
            gt_path = os.path.join(gt_dir, f"{pid}.txt")
            imwrite_safe(clean_path, img)
            if degrade:
                imwrite_safe(deg_path, degrade_page(img, seed=page_seed, level=level))
            else:
                deg_path = clean_path
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(gt)
            entry = {
                "id": pid, "source": pdf_path, "page": idx, "level": level,
                "real": not degrade, "seed": page_seed, "gt_chars": len(gt),
                "clean": os.path.relpath(clean_path, out_dir),
                "degraded": os.path.relpath(deg_path, out_dir),
                "gt": os.path.relpath(gt_path, out_dir),
            }
            if gt_validity:
                from doc_metrics import devanagari_validity
                audit = devanagari_validity(gt)
                entry["gt_invalid_tokens"] = audit["invalid_tokens"]
                entry["gt_invalid_token_rate"] = audit["invalid_token_rate"]
            entries.append(entry)
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

_COL_LEFT_ITEMS = ["Statement No: {n}", "Date: {d}", "Bill To:",
                   "ACME Supplies Ltd", "12 Market Street", "Lakeside 45000",
                   "GSTIN: 27AABCU9603R1ZM"]
_COL_RIGHT_ITEMS = ["Ship To:", "Northwind Traders", "88 Harbor Road",
                    "Riverside 56001", "Payment Terms: Net 30",
                    "Due Date: {due}", "Reference: PO-{po}"]


def render_two_column_document(seed: int = 0, dpi: int = 300) -> Tuple[np.ndarray, str]:
    """Two-column statement: full-width title, column block, then a table block.

    Ground truth is the *reading order* (title, full left column, full right
    column, table, totals) — the layout where OCR detection order interleaves
    columns and XY-cut ordering must recover column-major order.
    """
    rng = np.random.default_rng(seed)
    s = dpi / 150.0
    w, h = int(1240 * s), int(1754 * s)

    def sc(*vals):
        return tuple(int(round(v * s)) for v in vals)

    page = Image.new("RGB", (w, h), (250, 249, 246))
    d = ImageDraw.Draw(page)
    f_title = _font(int(44 * s))
    f_head = _font(int(24 * s))
    f_body = _font(int(22 * s))

    lines: List[str] = []
    title = "STATEMENT OF ACCOUNT"
    d.text(sc(80, 70), title, font=f_title, fill=(15, 15, 15))
    lines.append(title)

    inv_no = int(rng.integers(10000, 99999))
    po = int(rng.integers(1000, 9999))
    date = f"2026-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}"
    due = f"2026-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}"

    y0 = 200
    step = int(40 * s)
    left = [t.format(n=inv_no, d=date) if "{" in t else t for t in _COL_LEFT_ITEMS]
    right = [t.format(due=due, po=po) if "{" in t else t for t in _COL_RIGHT_ITEMS]
    for i, text in enumerate(left):
        d.text(sc(80, y0 + i * step), text, font=f_body, fill=(25, 25, 25))
        lines.append(text)
    for i, text in enumerate(right):
        d.text(sc(700, y0 + i * step), text, font=f_body, fill=(25, 25, 25))
        lines.append(text)

    y_table = y0 + len(left) * step + int(80 * s)
    d.text(sc(80, y_table), "Description", font=f_head, fill=(0, 0, 0))
    d.text(sc(700, y_table), "Amount", font=f_head, fill=(0, 0, 0))
    d.line(sc(80, y_table + 34, 1160, y_table + 34), fill=(120, 120, 120),
           width=max(1, int(2 * s)))
    lines.append("Description Amount")

    total = 0.0
    row_y = y_table + int(56 * s)
    n_items = int(rng.integers(2, 4))
    for idx in [int(i) for i in rng.choice(len(_ITEMS), size=n_items, replace=False)]:
        name, qty, unit = _ITEMS[idx]
        amount = qty * float(unit)
        total += amount
        d.text(sc(80, row_y), f"{name} x{qty}", font=f_body, fill=(25, 25, 25))
        d.text(sc(980, row_y), f"{amount:.2f}", font=f_body, fill=(25, 25, 25))
        lines.append(f"{name} x{qty} {amount:.2f}")
        row_y += int(40 * s)

    grand = round(total * 1.13, 2)
    d.text(sc(700, row_y + 30), "TOTAL", font=f_title, fill=(10, 10, 10))
    d.text(sc(980, row_y + 34), f"{grand:.2f}", font=f_title, fill=(10, 10, 10))
    lines.append(f"TOTAL {grand:.2f}")

    arr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    return arr, normalize_text("\n".join(lines))


def build_synthetic_dataset(out_dir: str, n: int = 20,
                            levels=("mild", "medium"), seed: int = 100,
                            layout: str = "single") -> Dict:
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    renderer = render_two_column_document if layout == "two_column" else render_synthetic_invoice
    for i in range(n):
        img, gt = renderer(seed)
        level = levels[i % len(levels)]
        page_seed = seed * 1000 + i
        pid = f"{'two' if layout == 'two_column' else 'inv'}_{i:04d}"
        clean_path = os.path.join(pages_dir, f"{pid}_clean.png")
        deg_path = os.path.join(pages_dir, f"{pid}_degraded.png")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        imwrite_safe(clean_path, img)
        imwrite_safe(deg_path, degrade_page(img, seed=page_seed, level=level))
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
                "layout": layout,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# synthetic Devanagari (Nepali / Hindi) documents
# ---------------------------------------------------------------------------

_DEVA_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\mangal.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
)

# Text needs complex-script shaping (conjuncts, matras) - PIL cannot do it, so
# the fixture renders through Qt (HarfBuzz), offscreen. Verified: QTextLayout
# reports shaped glyph runs (e.g. 'क्ष' 3 codepoints -> 1 glyph).
_DEVA_STRINGS = {
    "ne": {
        "org": "नेपाल सरकार",
        "dept": "आन्तरिक राजस्व कार्यालय, काठमाडौं",
        "doc": "बीजक नं",
        "date": "मिति",
        "to": "ग्राहकको नाम",
        "to_val": "हिमालय ट्रेडर्स प्रा. लि.",
        "addr": "पोखरा, कास्की",
        "head": ("विवरण", "संख्या", "दर", "रकम"),
        "items": (
            ("कापी", "२", "८५.००", "१७०.००"),
            ("कलम", "५", "२५.००", "१२५.००"),
            ("झोला", "१", "४५०.००", "४५०.००"),
        ),
        "subtotal": "उप-जम्मा",
        "vat": "मूल्य अभिवृद्धि कर",
        "total": "कुल जम्मा",
        "thanks": "धन्यवाद",
    },
    "hi": {
        "org": "भारत सरकार",
        "dept": "आयकर विभाग, नई दिल्ली",
        "doc": "बीजक संख्या",
        "date": "दिनांक",
        "to": "ग्राहक का नाम",
        "to_val": "श्री गणेश ट्रेडर्स",
        "addr": "जयपुर, राजस्थान",
        "head": ("विवरण", "मात्रा", "दर", "राशि"),
        "items": (
            ("पुस्तक", "३", "१२०.००", "३६०.००"),
            ("कागज", "२", "९०.००", "१८०.००"),
            ("स्याही", "१", "२५०.००", "२५०.००"),
        ),
        "subtotal": "उप-योग",
        "vat": "मूल्य वर्धित कर",
        "total": "कुल योग",
        "thanks": "धन्यवाद",
    },
}


def _qt_app():
    """Offscreen Qt app for shaped text rendering (dev fixture generation)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    return QGuiApplication.instance() or QGuiApplication([])


def _deva_font(size_px: int):
    from PySide6.QtGui import QFont, QFontDatabase
    for path in _DEVA_FONT_CANDIDATES:
        if os.path.exists(path):
            fid = QFontDatabase.addApplicationFont(path)
            fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
            if fams:
                font = QFont(fams[0])
                font.setPixelSize(size_px)
                return font
    raise RuntimeError("no Devanagari font found (tried: %s)"
                       % ", ".join(_DEVA_FONT_CANDIDATES))


def render_devanagari_invoice(seed: int = 0, dpi: int = 300,
                              script: str = "ne") -> Tuple[np.ndarray, str]:
    """Devanagari invoice page (Nepali/Hindi) rendered through Qt shaping.

    GT is the exact drawn-string list in reading order (row-major: header,
    parties, table rows left-to-right, totals, footer). Amounts use Devanagari
    digits for the money metric's sake.
    """
    _qt_app()
    from PySide6.QtGui import QColor, QImage, QPainter

    t = _DEVA_STRINGS[script]
    s = dpi / 200.0
    w, h = int(1240 * s), int(1754 * s)
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor(252, 251, 248))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    def draw(x, y, text, size):
        p.setFont(_deva_font(int(size * s)))
        p.setPen(QColor(22, 22, 28))
        p.drawText(int(x * s), int(y * s), text)

    gt: List[str] = []
    rng = np.random.default_rng(seed)

    draw(60, 70, t["org"], 46)
    gt.append(t["org"])
    draw(60, 126, t["dept"], 30)
    gt.append(t["dept"])

    doc_no = f"{rng.integers(10000, 99999)}"
    draw(60, 210, f"{t['doc']}: {doc_no}", 30)
    gt.append(f"{t['doc']}: {doc_no}")
    draw(60, 256, f"{t['date']}: 2082-09-05", 30)
    gt.append(f"{t['date']}: 2082-09-05")

    draw(60, 340, f"{t['to']}:", 30)
    gt.append(f"{t['to']}:")
    draw(60, 386, t["to_val"], 30)
    gt.append(t["to_val"])
    draw(60, 432, t["addr"], 30)
    gt.append(t["addr"])

    cols = (60, 660, 850, 1000)
    y = 530
    table_top = y - 34
    for cx, head in zip(cols, t["head"]):
        draw(cx, y, head, 30)
    gt.extend(t["head"])
    y += 56
    for item in t["items"]:
        for cx, cell in zip(cols, item):
            draw(cx, y, cell, 30)
        gt.extend(item)
        y += 56
    table_bottom = y - 34

    # Light table rules: real invoices have them, and they measurably help the
    # detector keep adjacent cells apart (without them the DB detector merged
    # neighbouring table cells on this fixture and recognition went with it).
    from PySide6.QtCore import Qt as _Qt
    pen = p.pen()
    pen.setColor(QColor(120, 120, 125))
    pen.setWidth(2)
    p.setPen(pen)
    rows = [table_top] + [table_top + 56 * (i + 1) for i in range(len(t["items"]) + 1)]
    for ry in rows:
        p.drawLine(int(60 * s), int(ry * s), int(1080 * s), int(ry * s))
    for cx in (60, 660, 850, 1000, 1080):
        p.drawLine(int(cx * s), int(table_top * s), int(cx * s), int(table_bottom * s))
    p.setPen(pen)

    y += 40
    for label, key in (("subtotal", "subtotal"), ("vat", "vat"), ("total", "total")):
        draw(660, y, t[key], 30)
        gt.append(t[key])
        y += 56
    draw(60, y + 80, t["thanks"], 36)
    gt.append(t["thanks"])

    p.end()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3]
    bgr = cv2.cvtColor(arr.copy().reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    return bgr, "\n".join(gt)


def build_devanagari_dataset(out_dir: str, n: int = 6, script: str = "ne",
                             levels=("mild", "medium", "heavy"), dpi: int = 300,
                             seed: int = 700) -> Dict:
    """Render + degrade Devanagari invoice pages with exact GT."""
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    for i in range(n):
        img, gt = render_devanagari_invoice(seed, dpi=dpi, script=script)
        level = levels[i % len(levels)]
        page_seed = seed * 1000 + i
        pid = f"deva_{script}_{i:04d}"
        clean_path = os.path.join(pages_dir, f"{pid}_clean.png")
        deg_path = os.path.join(pages_dir, f"{pid}_degraded.png")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        imwrite_safe(clean_path, img)
        imwrite_safe(deg_path, degrade_page(img, seed=page_seed, level=level))
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt)
        entries.append({
            "id": pid, "source": f"synthetic_{script}", "level": level,
            "seed": page_seed, "gt_chars": len(gt),
            "clean": os.path.relpath(clean_path, out_dir),
            "degraded": os.path.relpath(deg_path, out_dir),
            "gt": os.path.relpath(gt_path, out_dir),
        })
        seed += 1
    manifest = {"kind": "synthetic", "name": f"devanagari_{script}",
                "script": script, "dpi": dpi,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# mixed pages: text + logo + photo + signature (P5 router fixtures)
# ---------------------------------------------------------------------------

def _photo_texture(seed: int, size) -> np.ndarray:
    """Deterministic photo-like BGR texture for mixed-page fixtures.

    Prefers a center crop of a DIV2K image when available (real photographic
    content), else a smoothed noise field - tests must not depend on DIV2K.
    """
    w, h = size
    import glob as _glob
    hr = sorted(_glob.glob(os.path.join(BASE_DIR, "data", "DIV2K_valid_HR",
                                        "*.png")))
    if hr:
        img = imread_safe(hr[seed % len(hr)])
        if img is not None and img.shape[0] >= h and img.shape[1] >= w:
            y0 = (img.shape[0] - h) // 2
            x0 = (img.shape[1] - w) // 2
            return img[y0:y0 + h, x0:x0 + w].copy()
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), sigmaX=9)


def render_mixed_document(seed: int = 0, dpi: int = 300):
    """Mixed page: text + logo + photo + signature, exact text GT.

    Returns (image_bgr, gt_text, regions) where regions is
    [{"kind": "text"|"logo"|"photo"|"signature", "bbox": [x0,y0,x1,y1]}].
    Photo/logo/signature carry no text GT (by design - the router must keep
    them without inventing anything).
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

    lines: List[str] = []
    regions: List[Dict] = []

    d.text(sc(80, 70), "INVOICE", font=f_title, fill=(15, 15, 15))
    lines.append("INVOICE")

    logo_bbox = sc(940, 60, 1160, 200)
    d.ellipse(logo_bbox, fill=(30, 90, 200), outline=(10, 40, 110),
              width=max(1, int(3 * s)))
    d.rectangle(sc(1000, 110, 1120, 150), fill=(240, 200, 40))
    regions.append({"kind": "logo", "bbox": list(logo_bbox)})

    vendor = _VENDORS[int(rng.integers(0, len(_VENDORS)))]
    inv_no = int(rng.integers(10000, 99999))
    d.text(sc(80, 150), f"Vendor: {vendor}", font=f_head, fill=(30, 30, 30))
    d.text(sc(80, 185), f"Invoice No: {inv_no}", font=f_body, fill=(30, 30, 30))
    lines += [f"Vendor: {vendor}", f"Invoice No: {inv_no}"]

    photo_w, photo_h = int(360 * s), int(260 * s)
    photo_x, photo_y = int(800 * s), int(300 * s)
    texture = _photo_texture(seed, (photo_w, photo_h))
    page.paste(Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)),
               (photo_x, photo_y))
    d.rectangle(sc(800, 300, 1160, 560), outline=(90, 90, 90),
                width=max(1, int(2 * s)))
    regions.append({"kind": "photo",
                    "bbox": [photo_x, photo_y, photo_x + photo_w,
                             photo_y + photo_h]})

    table_y = int(640 * s)
    d.text(sc(80, table_y), "Description", font=f_head, fill=(0, 0, 0))
    d.text(sc(760, table_y), "Amount", font=f_head, fill=(0, 0, 0))
    lines.append("Description Amount")
    total = 0.0
    y = table_y + int(50 * s)
    n_items = int(rng.integers(2, 5))
    picked = [int(i) for i in rng.choice(len(_ITEMS), size=n_items,
                                         replace=False)]
    for idx in picked:
        name, qty, unit = _ITEMS[idx]
        amount = (int(qty) + int(rng.integers(0, 3))) * float(unit)
        total += amount
        d.text((int(80 * s), y), name, font=f_body, fill=(25, 25, 25))
        d.text((int(760 * s), y), f"{amount:.2f}", font=f_body, fill=(25, 25, 25))
        lines.append(f"{name} {amount:.2f}")
        y += int(44 * s)
    grand = round(total * 1.13, 2)
    d.text((int(700 * s), y + int(40 * s)), "TOTAL", font=f_title,
           fill=(10, 10, 10))
    d.text((int(920 * s), y + int(40 * s)), f"{grand:.2f}", font=f_title,
           fill=(10, 10, 10))
    lines.append(f"TOTAL {grand:.2f}")

    sig_bbox = sc(120, 1180, 560, 1360)
    pts = [(sig_bbox[0] + int(20 * s), sig_bbox[3] - int(60 * s))]
    for k in range(6):
        pts.append((sig_bbox[0] + int((40 + k * 70) * s),
                    sig_bbox[1] + int((60 + (k % 3) * 40) * s)))
    pts.append((sig_bbox[2] - int(20 * s), sig_bbox[3] - int(40 * s)))
    d.line(pts, fill=(20, 20, 60), width=max(1, int(3 * s)), joint="curve")
    regions.append({"kind": "signature", "bbox": list(sig_bbox)})

    text_bbox = sc(60, 40, 1180, 1500)
    regions.append({"kind": "text", "bbox": list(text_bbox)})

    arr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    return arr, normalize_text("\n".join(lines)), regions


def build_mixed_dataset(out_dir: str, n: int = 6, dpi: int = 300,
                        seed: int = 100) -> Dict:
    """Mixed pages with exact text GT + region sidecars (P5 router set)."""
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    for i in range(n):
        img, gt, regions = render_mixed_document(seed=seed + i, dpi=dpi)
        pid = f"mixed_{i:04d}"
        clean_path = os.path.join(pages_dir, f"{pid}_clean.png")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        regions_path = os.path.join(gt_dir, f"{pid}.regions.json")
        imwrite_safe(clean_path, img)
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt)
        with open(regions_path, "w", encoding="utf-8") as f:
            json.dump(regions, f, ensure_ascii=False, indent=1)
        entries.append({
            "id": pid, "source": "synthetic_mixed", "seed": seed + i,
            "gt_chars": len(gt), "kinds": sorted({r["kind"] for r in regions}),
            "clean": os.path.relpath(clean_path, out_dir),
            "degraded": os.path.relpath(clean_path, out_dir),
            "gt": os.path.relpath(gt_path, out_dir),
            "regions": os.path.relpath(regions_path, out_dir),
        })
    manifest = {"kind": "mixed_pages", "name": "mixed_pages", "dpi": dpi,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entries": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_photo_proxy_dataset(src_data_dir: str, out_dir: str,
                              level: str = "heavy", max_side: int = 1700,
                              seed: int = 1234,
                              gt_dir: Optional[str] = None) -> Dict:
    """Phone-photo proxy: degrade an existing page set (seeded, deterministic).

    `gt_dir` optionally replaces the source GT files (matched by entry id) -
    e.g. the Gemini anchor texts, since modern-PDF text layers are corrupt.
    This is a *proxy* for the A5 field set (real photographs), labeled as
    such: photo artifacts come from `degradation_document.degrade_page`.
    """
    src = load_dataset(src_data_dir)
    pages_dir = os.path.join(out_dir, "pages")
    gt_out = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)
    entries: List[Dict] = []
    for i, e in enumerate(src["entries"]):
        img = imread_safe(e["_degraded_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        if max_side and max(h, w) > max_side:
            f = max_side / max(h, w)
            img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                             interpolation=cv2.INTER_AREA)
        degraded = degrade_page(img, seed=seed + i, level=level)
        pid = e["id"]
        page_path = os.path.join(pages_dir, f"{pid}_photo.png")
        imwrite_safe(page_path, degraded)
        gt_src = (os.path.join(gt_dir, f"{pid}.txt")
                  if gt_dir else e["_gt_path"])
        gt_text = open(gt_src, encoding="utf-8").read()
        gt_path = os.path.join(gt_out, f"{pid}.txt")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt_text)
        entries.append({
            "id": pid, "source": f"photo_proxy:{level}", "level": level,
            "seed": seed + i, "gt_chars": len(gt_text),
            "clean": os.path.relpath(page_path, out_dir),
            "degraded": os.path.relpath(page_path, out_dir),
            "gt": os.path.relpath(gt_path, out_dir),
        })
    manifest = {"kind": "photo_proxy", "name": f"photo_proxy_{level}",
                "level": level, "max_side": max_side, "seed": seed,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entries": entries}
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
        imwrite_safe(page_path, arr)
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


def build_line_crop_dataset(source: str, out_dir: str, max_lines: int = 0,
                            split: str = "test") -> Dict:
    """Real text-line crops + transcriptions (recognition-level eval).

    Used for the Nepali line slice (himalaya-ai/nepali-deva-ocr-eval, itself a
    held-out extraction of gauravgiri/nepali-ocr-dataset). Provenance and
    license are UNKNOWN: every artifact derived from this set must carry the
    'unverified provenance' label until a human eyeball pass clears it. The
    crops are recognition-only: detection is bypassed, so this measures the
    recognizer, not the page pipeline.
    """
    from datasets import load_dataset

    ds = load_dataset(source, split=split)
    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    for idx, example in enumerate(ds):
        if max_lines and len(entries) >= max_lines:
            break
        gt = str(example.get("ocr") or "").strip()
        img = example.get("image")
        if not gt or img is None:
            continue
        if isinstance(img, dict) and img.get("bytes"):
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(img["bytes"]))
        arr = np.array(img.convert("RGB"))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        pid = f"line_{idx:05d}"
        img_path = os.path.join(pages_dir, f"{pid}.png")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        imwrite_safe(img_path, arr)
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt)
        entries.append({
            "id": pid, "source": source, "row": idx, "real": True,
            "gt_chars": len(gt),
            "image": os.path.relpath(img_path, out_dir),
            "gt": os.path.relpath(gt_path, out_dir),
        })
    manifest = {
        "kind": "lines", "name": f"lines_{source.split('/')[-1]}",
        "source": source, "split": split,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_line_dataset(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for e in manifest["entries"]:
        e["_image_path"] = os.path.join(data_dir, e["image"])
        e["_gt_path"] = os.path.join(data_dir, e["gt"])
    return manifest


# ---------------------------------------------------------------------------
# heiDATA printed Devanagari (ALTO ground truth, CC BY 4.0)
# ---------------------------------------------------------------------------

def parse_alto_page(xml_bytes: bytes) -> Dict:
    """Parse one Transkribus-style ALTO v4 page.

    Returns {"width", "height", "lines": [{"text", "bbox"}...]} in page image
    pixel coordinates. Namespace-agnostic (v2/v3/v4 share the tag names).
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)

    def tag(el):
        return el.tag.split("}")[-1]

    page = None
    for el in root.iter():
        if tag(el) == "Page":
            page = el
            break
    if page is None:
        return {"width": 0, "height": 0, "lines": []}
    width = int(float(page.get("WIDTH", 0)))
    height = int(float(page.get("HEIGHT", 0)))
    lines = []
    for el in root.iter():
        if tag(el) != "TextLine":
            continue
        strings = [s.get("CONTENT", "") for s in el if tag(s) == "String"]
        text = " ".join(s for s in strings if s).strip()
        if not text:
            continue
        x, y = int(float(el.get("HPOS", 0))), int(float(el.get("VPOS", 0)))
        w, h = int(float(el.get("WIDTH", 0))), int(float(el.get("HEIGHT", 0)))
        lines.append({"text": text, "bbox": [x, y, x + w, y + h]})
    return {"width": width, "height": height, "lines": lines}


def build_heidata_dataset(zips_dir: str, out_dir: str,
                          max_pages_per_book: int = 0,
                          books: List[str] | None = None) -> Dict:
    """Real scanned Devanagari book pages + ALTO word/line ground truth.

    Source: heiDATA "Ground Truth data for printed Devanagari" (Merkel-Hilf
    2022, doi:10.11588/data/EGOKEI, CC BY 4.0) — Transkribus exports (jpg +
    ALTO v4) of letterpress books printed in Devanagari (Hindi/Sanskrit/Braj).
    Human-corrected transcription, unlike the gauravgiri line slice. This is
    the first page-level real set with boxes: GT boxes enable detection
    coverage, which CER alone cannot measure.
    """
    import zipfile

    pages_dir = os.path.join(out_dir, "pages")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    entries: List[Dict] = []
    for zip_path in sorted(glob.glob(os.path.join(zips_dir, "*.zip"))):
        book = os.path.splitext(os.path.basename(zip_path))[0]
        if books and book not in books:
            continue
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            jpgs = sorted(n for n in names if n.lower().endswith(".jpg"))
            if max_pages_per_book:
                jpgs = jpgs[:max_pages_per_book]
            for img_name in jpgs:
                stem = os.path.splitext(os.path.basename(img_name))[0]
                alto_name = next((n for n in names
                                  if "/alto/" in n and
                                  os.path.basename(n) == stem + ".xml"), None)
                if alto_name is None:
                    continue
                alto = parse_alto_page(z.read(alto_name))
                if not alto["lines"]:
                    continue
                import io
                from PIL import Image
                pil = Image.open(io.BytesIO(z.read(img_name))).convert("RGB")
                arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                ih, iw = arr.shape[:2]
                sx = iw / alto["width"] if alto["width"] else 1.0
                sy = ih / alto["height"] if alto["height"] else 1.0
                boxes = []
                for line in alto["lines"]:
                    x0, y0, x1, y1 = line["bbox"]
                    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                        x0, x1 = int(x0 * sx), int(x1 * sx)
                        y0, y1 = int(y0 * sy), int(y1 * sy)
                    boxes.append({"text": line["text"],
                                  "bbox": [x0, y0, x1, y1],
                                  "granularity": "line"})
                gt_text = "\n".join(b["text"] for b in boxes)
                pid = f"{book}_p{stem}"
                img_path = os.path.join(pages_dir, f"{pid}.png")
                gt_path = os.path.join(gt_dir, f"{pid}.txt")
                boxes_path = os.path.join(gt_dir, f"{pid}.boxes.json")
                imwrite_safe(img_path, arr)
                with open(gt_path, "w", encoding="utf-8") as f:
                    f.write(gt_text)
                with open(boxes_path, "w", encoding="utf-8") as f:
                    json.dump(boxes, f, ensure_ascii=False, indent=1)
                entries.append({
                    "id": pid, "source": "heidata:doi:10.11588/data/EGOKEI",
                    "book": book, "page": stem, "level": "real", "real": True,
                    "gt_chars": len(gt_text), "n_lines": len(boxes),
                    "clean": os.path.relpath(img_path, out_dir),
                    "degraded": os.path.relpath(img_path, out_dir),
                    "gt": os.path.relpath(gt_path, out_dir),
                    "boxes": os.path.relpath(boxes_path, out_dir),
                })
    manifest = {
        "kind": "real_pages", "name": "heidata_printed_devanagari",
        "source": "Merkel-Hilf 2022, doi:10.11588/data/EGOKEI (CC BY 4.0)",
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
        if e.get("boxes"):
            e["_boxes_path"] = os.path.join(data_dir, e["boxes"])
        if e.get("regions"):
            with open(os.path.join(data_dir, e["regions"]),
                      encoding="utf-8") as rf:
                e["_regions"] = json.load(rf)
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
    ap.add_argument("--layout", default="single", choices=["single", "two_column"],
                    help="synthetic page layout")
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
        sub = "synthetic" if args.layout == "single" else f"synthetic_{args.layout}"
        m = build_synthetic_dataset(os.path.join(args.out, sub),
                                    n=args.synthetic, levels=levels,
                                    seed=args.seed, layout=args.layout)
        print(f"synthetic[{args.layout}]: {len(m['entries'])} pages -> "
              f"{os.path.join(args.out, sub)}")
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
