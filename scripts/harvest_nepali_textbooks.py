"""
harvest_nepali_textbooks.py — textbook sources for the W-D real-print probe.

Two sources, both recorded in docs/LICENSES.md and never redistributed:

* `--source moest` (DSpace 6, elibrary.moest.gov.np): government reading
  materials. The shared harvest gate (`harvest_nepali_pdfs.probe_pdf` /
  `gate_reason`) decides whether the PDF has a clean Unicode text layer.
  Measured 2026-09-25: the sampled Nepali documents use legacy non-Unicode
  fonts (Devanagari ratio <= 0.02, invalid sequences 7-8%) and are rejected -
  recorded as negative evidence, not silently dropped.
* `--source cornell` (DSpace 7, ecommons.cornell.edu, collection 1813/24179):
  458 Nepali textbook scans. Their DSpace TEXT derivatives are 112-276 bytes
  (verified), i.e. no text layer at all; they are downloaded as an *unlabeled*
  real-print probe for `scripts/profile_textbook_scans.py` (behaviour/timing
  only, no CER claims).

The CDC catalogue (`lib.moecdc.gov.np/elibrary`) lists ~90 e-copies, but its
ResourceSpace download endpoints return 404 with and without a session
(2026-09-25), so it is not a source.

Usage:
    python scripts/harvest_nepali_textbooks.py --source moest --limit 20
    python scripts/harvest_nepali_textbooks.py --source cornell --limit 5
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harvest_nepali_pdfs import gate_reason, probe_pdf  # noqa: E402

MOEST_BASE = "http://elibrary.moest.gov.np:8080"
MOEST_QUERIES = ["textbook", "curriculum", "teacher guide", "reading material"]
CORNELL_API = "https://ecommons.cornell.edu/server/api"
CORNELL_QUERY = "Nepali textbooks"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) VeriScript-Research/1.0 "
      "(internal OCR evaluation; contact: local)")

_HANDLE_RE = re.compile(r'href="(/handle/\d+/\d+)"')
_META_RE = re.compile(
    r'<meta\s+name="(citation_title|citation_pdf_url)"\s+content="([^"]*)"')


def _get(url: str, timeout: int = 90) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _download(url: str, dst: str, retries: int = 3) -> None:
    """Atomic download (..part -> replace), PDF magic verified, no partials."""
    part = dst + ".part"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as resp, \
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
    try:
        os.remove(part)
    except OSError:
        pass
    raise RuntimeError(f"download failed after {retries}: {last_err}")


def parse_moest_search(page: str) -> List[str]:
    """Unique `/handle/...` paths from a DSpace 6 simple-search page."""
    seen, out = set(), []
    for m in _HANDLE_RE.finditer(page):
        h = m.group(1)
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def parse_moest_item(page: str) -> Dict:
    """Title + `citation_pdf_url` from a DSpace 6 item page (may be empty)."""
    out: Dict[str, str] = {"title": "", "pdf_url": ""}
    for name, value in _META_RE.findall(page):
        if name == "citation_title" and not out["title"]:
            out["title"] = html_mod.unescape(value).strip()
        elif name == "citation_pdf_url" and not out["pdf_url"]:
            out["pdf_url"] = html_mod.unescape(value).strip()
    return out


def parse_cornell_items(payload: Dict) -> List[Dict]:
    """(uuid, handle, title) for each item in a DSpace 7 search response."""
    result = payload.get("_embedded", {}).get("searchResult", {})
    out = []
    for obj in result.get("_embedded", {}).get("objects", []):
        item = obj.get("_embedded", {}).get("indexableObject", {})
        meta = item.get("metadata", {})
        title = (meta.get("dc.title") or [{}])[0].get("value", "")
        out.append({"uuid": item.get("uuid", ""), "handle": item.get("handle", ""),
                    "title": title})
    return out


def pick_cornell_pdf(bundles: Dict) -> Optional[Dict]:
    """First ORIGINAL PDF bitstream (content URL) in a DSpace 7 item.

    `?embed=bitstreams` nests the list one level deeper
    (`bundle._embedded.bitstreams._embedded.bitstreams`); both shapes work.
    """
    for bundle in bundles.get("_embedded", {}).get("bundles", []):
        if bundle.get("name") != "ORIGINAL":
            continue
        bits = bundle.get("_embedded", {}).get("bitstreams", [])
        if isinstance(bits, dict):
            bits = bits.get("_embedded", {}).get("bitstreams", [])
        for bit in bits:
            name = bit.get("name", "")
            if not name.lower().endswith(".pdf"):
                continue
            return {"name": name, "bytes": bit.get("sizeBytes", 0),
                    "url": bit.get("_links", {}).get("content", {}).get("href", "")}
    return None


def harvest_moest(out_dir: str, limit: int = 20,
                  queries: Optional[List[str]] = None) -> Dict:
    """Search -> item pages -> PDFs -> text-layer gate. Rejects are deleted."""
    os.makedirs(out_dir, exist_ok=True)
    report = {"source": "moest", "accepted": [], "rejected": [], "errors": [],
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    handles: List[str] = []
    for query in queries or MOEST_QUERIES:
        try:
            page = _get(f"{MOEST_BASE}/simple-search?query="
                        + urllib.request.quote(query))
        except Exception as e:  # noqa: BLE001
            report["errors"].append({"query": query, "error": str(e)})
            continue
        for handle in parse_moest_search(page):
            if handle not in handles:
                handles.append(handle)
        if len(handles) >= limit:
            break
    print(f"[moest] {len(handles)} candidate items")
    for handle in handles[:limit]:
        try:
            info = parse_moest_item(_get(MOEST_BASE + handle))
        except Exception as e:  # noqa: BLE001
            report["errors"].append({"handle": handle, "error": str(e)})
            continue
        if not info["pdf_url"]:
            report["errors"].append({"handle": handle, "title": info["title"],
                                     "error": "no citation_pdf_url"})
            continue
        name = f"moest_{handle.rsplit('/', 1)[-1]}.pdf"
        dst = os.path.join(out_dir, name)
        rec = {"handle": handle, "title": info["title"], "file": name,
               "url": info["pdf_url"]}
        try:
            if not os.path.exists(dst):
                _download(info["pdf_url"], dst)
            probe = probe_pdf(dst)
            reason = gate_reason(probe)
            rec.update(probe)
            rec["reason"] = reason
            if reason is None:
                rec["verdict"] = "accepted"
                report["accepted"].append(rec)
                print(f"  OK   {name}: {probe['pages']}p "
                      f"{probe['chars_per_page']:.0f} chars/p "
                      f"deva {probe['devanagari_ratio']:.2f}")
            else:
                rec["verdict"] = "rejected"
                report["rejected"].append(rec)
                os.remove(dst)
                print(f"  DROP {name}: {reason}")
        except Exception as e:  # noqa: BLE001
            report["errors"].append({**rec, "error": str(e)})
            print(f"  ERR  {name}: {e}")
    return report


def harvest_cornell(out_dir: str, limit: int = 5,
                    query: str = CORNELL_QUERY) -> Dict:
    """Download the first ORIGINAL PDF of `limit` real textbook scans."""
    os.makedirs(out_dir, exist_ok=True)
    report = {"source": "cornell", "downloaded": [], "errors": [],
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    url = (f"{CORNELL_API}/discover/search/objects?"
           f"query={urllib.request.quote(query)}&dsoType=item&size={limit * 2}")
    items = parse_cornell_items(json.loads(_get(url)))
    print(f"[cornell] {len(items)} candidate items")
    for item in items:
        if len(report["downloaded"]) >= limit:
            break
        try:
            bundles = json.loads(_get(
                f"{CORNELL_API}/core/items/{item['uuid']}/bundles"
                f"?embed=bitstreams"))
            pdf = pick_cornell_pdf(bundles)
            if not pdf:
                continue
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", item["handle"].replace("/", "_"))
            name = f"cornell_{stem}.pdf"
            dst = os.path.join(out_dir, name)
            if not os.path.exists(dst):
                _download(pdf["url"], dst)
            rec = {"handle": item["handle"], "title": item["title"],
                   "file": name, "bitstream": pdf["name"],
                   "bytes": os.path.getsize(dst)}
            report["downloaded"].append(rec)
            print(f"  OK   {name}: {rec['bytes'] / 1e6:.1f} MB | {item['title']}")
        except Exception as e:  # noqa: BLE001
            report["errors"].append({**item, "error": str(e)})
            print(f"  ERR  {item.get('handle')}: {e}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Harvest textbook PDFs (W-D probe)")
    ap.add_argument("--source", choices=["moest", "cornell"], required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.source == "moest":
        out = args.out or os.path.join(BASE_DIR, "data", "doc_eval",
                                       "nepali_moest_sources")
        report = harvest_moest(out, args.limit)
    else:
        out = args.out or os.path.join(BASE_DIR, "data", "doc_eval",
                                       "nepali_textbook_scans")
        report = harvest_cornell(out, args.limit)
    print(f"[harvest] source={args.source} -> {out}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[harvest] wrote {args.json}")


if __name__ == "__main__":
    main()
