"""
harvest_heidata_books.py — fetch the unused heiDATA Devanagari books.

The heiDATA dataset "Ground Truth data for printed Devanagari"
(doi:10.11588/data/EGOKEI, CC BY 4.0) holds 19 letterpress books with
human-corrected Transkribus ALTO ground truth. Eleven were used for the
frozen eval sets (7 books) and dev tuning (4 books); the remaining books are
training-only material for the line recognizer (W1 attempt 3).

Downloads are resumable (existing valid zips are skipped), validated after
fetch, and recorded with size + sha256 in `harvest.json` next to the zips.

Usage:
    python scripts/harvest_heidata_books.py --list
    python scripts/harvest_heidata_books.py --out data/doc_eval/heidata/zips
    python scripts/harvest_heidata_books.py --only vyasa1906,vyasa1896
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

API_DATASET = ("https://heidata.uni-heidelberg.de/api/datasets/:persistentId/"
               "?persistentId=doi:10.11588/data/EGOKEI")
API_FILE = "https://heidata.uni-heidelberg.de/api/access/datafile/{}"
# The hosting WAF challenges browser-like clients; the API answers plain curl.
UA = "curl/8.5.0"

# Books already used by the frozen eval sets and the dev set (W1 attempts 1-2).
USED_BOOKS = {
    "diksita1895", "dudhadasa1900", "hajarilala1919", "jacobi1897",
    "jagannatha1955", "jayadeva1926", "jnanadasa1895", "paramananda1924",
    "pyarelala1914", "sankaracarya1925", "sivaramasukla1900",
}
# Held out as a NEW frozen secondary set (never trained on).
HOLDOUT_BOOKS = {"saktidharasukla1930", "simha1914"}


def list_files() -> List[Dict]:
    req = urllib.request.Request(API_DATASET, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    out = []
    for f in data["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        out.append({"id": df["id"], "name": df["filename"],
                    "size": df.get("filesize", 0)})
    return sorted(out, key=lambda x: x["name"])


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def valid_book_zip(path: str) -> bool:
    """A usable book has page images and ALTO XML and is not truncated."""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return False
            names = z.namelist()
            return (any(n.lower().endswith(".jpg") for n in names)
                    and any("/alto/" in n and n.endswith(".xml") for n in names))
    except (zipfile.BadZipFile, OSError):
        return False


def download(file_id: int, dest: str, expected_size: int = 0) -> None:
    tmp = dest + ".part"
    req = urllib.request.Request(API_FILE.format(file_id),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp, \
            open(tmp, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    if expected_size and os.path.getsize(tmp) != expected_size:
        raise RuntimeError(f"size mismatch: {os.path.getsize(tmp)} != "
                           f"{expected_size}")
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description="Download heiDATA books")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "doc_eval",
                                                  "heidata", "zips"))
    ap.add_argument("--only", default=None,
                    help="comma-separated book stems (default: all unused)")
    ap.add_argument("--list", action="store_true", help="list remote files")
    args = ap.parse_args()

    files = list_files()
    if args.list:
        have = set(os.listdir(args.out)) if os.path.isdir(args.out) else set()
        for f in files:
            state = "have" if f["name"] in have else (
                "used" if f["name"][:-4] in USED_BOOKS else "NEW")
            print(f'  {f["name"]:42s} {f["size"] / 1e6:7.1f} MB  [{state}]')
        return

    os.makedirs(args.out, exist_ok=True)
    wanted = None
    if args.only:
        wanted = {b.strip() for b in args.only.split(",") if b.strip()}
    record_path = os.path.join(os.path.dirname(args.out), "harvest.json")
    record: Dict = {"dataset": "doi:10.11588/data/EGOKEI", "books": {}}
    if os.path.exists(record_path):
        record = json.load(open(record_path, encoding="utf-8"))

    todo = []
    for f in files:
        stem = f["name"][:-4]
        if stem in USED_BOOKS and not args.only:
            continue
        if wanted is not None and stem not in wanted:
            continue
        todo.append(f)

    for k, f in enumerate(todo, 1):
        dest = os.path.join(args.out, f["name"])
        if os.path.exists(dest) and valid_book_zip(dest):
            print(f"  [{k}/{len(todo)}] {f['name']}: already present, valid")
        else:
            t0 = time.time()
            print(f"  [{k}/{len(todo)}] {f['name']}: downloading "
                  f"({f['size'] / 1e6:.1f} MB)...", flush=True)
            download(f["id"], dest, expected_size=f["size"])
            print(f"      done in {time.time() - t0:.0f}s", flush=True)
        record["books"][f["name"]] = {
            "id": f["id"], "size": os.path.getsize(dest),
            "sha256": sha256_file(dest),
            "role": "holdout" if f["name"][:-4] in HOLDOUT_BOOKS else "train",
        }
        with open(record_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
    print(f"record: {record_path}")
    total = sum(v["size"] for v in record["books"].values())
    print(f"books recorded: {len(record['books'])} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
