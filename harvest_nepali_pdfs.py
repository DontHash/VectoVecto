"""
harvest_nepali_pdfs.py — collect born-digital Nepali PDFs for page-level eval.

Why: synthetic fixtures and letterpress books are not today's Nepali documents.
Government publications are born-digital (real fonts, layouts, modern Nepali)
and their text layer can serve as ground truth — the same method as the arXiv
set, but in Devanagari. The catch is text layers: many Nepali PDFs are scans
(no text) or use legacy non-Unicode fonts (Preeti etc.), which extract as
mojibake. Both are rejected here by a Devanagari-ratio gate.

Accepted PDFs go to data/doc_eval/nepali_pdf_sources/ (gitignored). The
resulting dataset is internal-eval-only; provenance recorded per file.

Usage:
    python harvest_nepali_pdfs.py --out data/doc_eval/nepali_pdf_sources \
        --json out/nepali_pdf_harvest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Candidate URLs found via agent-reach search (Exa). A mix of statute PDFs and
# court judgments; all public government publications (Nepal). Verified below.
CANDIDATES = [
    "https://lawcommission.gov.np/np/wp-content/uploads/2021/08/"
    "%E0%A4%AE%E0%A5%81%E0%A4%B2%E0%A5%81%E0%A4%95%E0%A5%80-"
    "%E0%A4%90%E0%A4%A8-%E0%A5%A7%E0%A5%AF%E0%A5%A7%E0%A5%A6.pdf",
    "https://supremecourt.gov.np/web/assets/downloads/judgements/supreme_218512.pdf",
    "https://supremecourt.gov.np/court/public/media/2024_05/"
    "40a6182ed9d25964d02378e1442d7073.pdf",
    "https://supremecourt.gov.np/court/public/media/2024_08/"
    "dcb178d233d58e0d17d78150fb44f28a.pdf",
    "https://elibrary.moest.gov.np:8080/bitstream/123456789/99/1/"
    "%E0%A4%85%E0%A4%A8%E0%A4%BF%E0%A4%B5%E0%A4%BE%E0%A4%B0%E0%A5%8D%E0%A4%AF-"
    "%E0%A4%A4%E0%A4%A5%E0%A4%BE-"
    "%E0%A4%A8%E0%A4%BF%E0%A4%83%E0%A4%B6%E0%A5%81%E0%A4%B2%E0%A5%8D%E0%A4%95-"
    "%E0%A4%B6%E0%A4%BF%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%BE-"
    "%E0%A4%B8%E0%A4%AE%E0%A5%8D%E0%A4%AC%E0%A4%A8%E0%A5%8D%E0%A4%A7%E0%A5%80-"
    "%E0%A4%A8%E0%A4%BF%E0%A4%AF%E0%A4%AE%E0%A4%BE%E0%A4%B5%E0%A4%B2%E0%A5%80-"
    "%E0%A5%A8%E0%A5%A6%E0%A5%AD%E0%A5%AD.pdf",
    "https://supremecourt.gov.np/web/assets/downloads/judgements/"
    "Some%20Landmark%20Decision%20-%20Vol.%203.pdf",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) VectorScaling-Research/1.0 "
      "(internal OCR evaluation; contact: local)")


def _devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    deva = sum(1 for c in letters if "\u0900" <= c <= "\u097f")
    return deva / len(letters)


def probe_pdf(path: str, max_pages: int = 8) -> dict:
    """Text-layer quality of a PDF: chars/pages + Devanagari ratio (pre-mojibake)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    n_pages = len(doc)
    texts = []
    for i in range(min(n_pages, max_pages)):
        texts.append(doc[i].get_textpage().get_text_range())
    text = "\n".join(texts)
    return {
        "pages": n_pages,
        "sampled_pages": len(texts),
        "chars": len(text),
        "chars_per_page": len(text) / len(texts) if texts else 0,
        "devanagari_ratio": round(_devanagari_ratio(text), 3),
    }


def _download(url: str, dst: str, retries: int = 3) -> None:
    """Atomic download (..part -> replace) with retries; never leaves partials."""
    import shutil
    part = dst + ".part"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp, \
                    open(part, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 16)
            with open(part, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("not a PDF response")
            os.replace(part, dst)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries}: {last_err}")


def discover_pdf_links(page_urls, limit_per_page: int = 4):
    """Collect .pdf hrefs from listing/content pages (absolute URLs)."""
    import re
    out = []
    for page in page_urls:
        try:
            req = urllib.request.Request(page, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                html = resp.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"  [discover] {page}: {e}")
            continue
        found = []
        for m in re.finditer(r'href="([^"]+\.pdf)"', html, re.I):
            found.append(urllib.parse.urljoin(page, m.group(1)))
        seen = set()
        uniq = [u for u in found if not (u in seen or seen.add(u))]
        print(f"  [discover] {page}: {len(uniq)} pdf links")
        out.extend(uniq[:limit_per_page])
    return out


def harvest(out_dir: str, urls=None, min_chars_per_page: int = 300,
            min_deva_ratio: float = 0.4) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    report = {"accepted": [], "rejected": [], "created":
              time.strftime("%Y-%m-%d %H:%M:%S")}
    for url in (urls or CANDIDATES):
        name = os.path.basename(urllib.parse.unquote(url.split("?")[0]))
        if not name.lower().endswith(".pdf"):
            name = f"doc_{abs(hash(url)) % 10**8}.pdf"
        dst = os.path.join(out_dir, name)
        rec = {"url": url, "file": name}
        try:
            if os.path.exists(dst):
                try:
                    info = probe_pdf(dst)
                except Exception:  # noqa: BLE001
                    os.remove(dst)
                    _download(url, dst)
                    info = probe_pdf(dst)
            else:
                _download(url, dst)
                info = probe_pdf(dst)
            rec.update(info)
            if (info["chars_per_page"] >= min_chars_per_page and
                    info["devanagari_ratio"] >= min_deva_ratio):
                rec["verdict"] = "accepted"
                report["accepted"].append(rec)
                print(f"  OK   {name}: {info['pages']}p "
                      f"{info['chars_per_page']:.0f} chars/p "
                      f"deva {info['devanagari_ratio']:.2f}")
            else:
                rec["verdict"] = "rejected"
                report["rejected"].append(rec)
                os.remove(dst)
                print(f"  DROP {name}: {info['chars_per_page']:.0f} chars/p "
                      f"deva {info['devanagari_ratio']:.2f} (scan or mojibake)")
        except Exception as e:  # noqa: BLE001
            rec["verdict"] = "error"
            rec["error"] = str(e)
            report["rejected"].append(rec)
            print(f"  ERR  {name}: {e}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Harvest Nepali born-digital PDFs")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "doc_eval",
                                                  "nepali_pdf_sources"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-chars-per-page", type=int, default=300)
    ap.add_argument("--min-deva-ratio", type=float, default=0.4)
    ap.add_argument("--discover", nargs="*", default=[],
                    help="listing/content pages to scrape for .pdf links")
    args = ap.parse_args()
    urls = list(CANDIDATES)
    if args.discover:
        urls += discover_pdf_links(args.discover)
    report = harvest(args.out, urls=urls,
                     min_chars_per_page=args.min_chars_per_page,
                     min_deva_ratio=args.min_deva_ratio)
    print(f"[harvest] accepted {len(report['accepted'])} / "
          f"rejected {len(report['rejected'])}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[harvest] wrote {args.json}")


if __name__ == "__main__":
    main()
