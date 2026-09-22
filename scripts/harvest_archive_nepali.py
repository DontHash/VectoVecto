"""
harvest_archive_nepali.py — unlabeled real Nepali scans from archive.org.

These pages have no reliable transcription (archive.org's own OCR is not a
Devanagari GT), so they are used for *behavior* checks only: orientation
decisions, reading-order stability, review-queue rates and latency on real
scans — see eval_sanity.py. No CER may be quoted from this set.

Usage:
    python scripts/harvest_archive_nepali.py --json out/nepali_unlabeled_harvest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402

# Public archive.org items (mixed book / newspaper / handwritten inventory).
ITEMS = [
    "MunaMadankoAgrajJiWayalaLachchhiMadduniByPushpaChitrakar",
    "goldennewspkr_gmail_20151011",
    "lisma-dhala-2011",
    "goldennewspkr_gmail_20151010",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) VectoVecto-Research/1.0 "
      "(internal OCR evaluation)")


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _pick_pdf(files) -> str | None:
    for f in files:
        if f.get("format") in ("Text PDF", "Additional Text PDF"):
            return f.get("name")
    for f in files:
        if str(f.get("name", "")).lower().endswith(".pdf"):
            return f.get("name")
    return None


def harvest(out_dir: str, items=None, pages_per_item: int = 6, dpi: int = 300,
            overwrite: bool = False) -> dict:
    import pypdfium2 as pdfium

    os.makedirs(out_dir, exist_ok=True)
    report = {"items": [], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    for ident in (items or ITEMS):
        rec = {"identifier": ident, "pages": []}
        try:
            meta = _get_json(f"https://archive.org/metadata/{ident}")
            pdf_name = _pick_pdf(meta.get("files", []))
            if not pdf_name:
                rec["error"] = "no pdf in item"
                report["items"].append(rec)
                print(f"  DROP {ident}: no pdf")
                continue
            pdf_path = os.path.join(out_dir, f"{ident}.pdf")
            if overwrite or not os.path.exists(pdf_path):
                import urllib.parse
                url = ("https://archive.org/download/"
                       f"{urllib.parse.quote(ident)}/"
                       f"{urllib.parse.quote(pdf_name)}")
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as resp, \
                        open(pdf_path + ".part", "wb") as f:
                    while True:
                        block = resp.read(1 << 16)
                        if not block:
                            break
                        f.write(block)
                os.replace(pdf_path + ".part", pdf_path)
            doc = pdfium.PdfDocument(pdf_path)
            n = min(len(doc), pages_per_item)
            for i in range(n):
                pid = f"{ident}_p{i:03d}"
                img_path = os.path.join(out_dir, f"{pid}.png")
                if overwrite or not os.path.exists(img_path):
                    page = doc[i]
                    pil = page.render(scale=dpi / 72.0).to_pil().convert("RGB")
                    import numpy as np
                    import cv2
                    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    doc_data.imwrite_safe(img_path, arr)
                rec["pages"].append({"id": pid, "image": f"{pid}.png"})
            print(f"  OK   {ident}: {n} pages ({pdf_name})")
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
            print(f"  ERR  {ident}: {e}")
        report["items"].append(rec)
    entries = []
    for rec in report["items"]:
        for p in rec.get("pages", []):
            entries.append({"id": p["id"], "image": p["image"],
                            "unlabeled": True})
    manifest = {"kind": "unlabeled", "name": "nepali_unlabeled",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "archive.org public items", "entries": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Harvest unlabeled Nepali scans")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "doc_eval",
                                                  "nepali_unlabeled"))
    ap.add_argument("--pages-per-item", type=int, default=6)
    ap.add_argument("--json", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    report = harvest(args.out, pages_per_item=args.pages_per_item,
                     overwrite=args.overwrite)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[harvest] wrote {args.json}")


if __name__ == "__main__":
    main()
