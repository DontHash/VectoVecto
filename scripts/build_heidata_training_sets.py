"""
build_heidata_training_sets.py — book-level split for W1 attempt 3.

Six unused heiDATA books become training data (`heidata_train`); two are held
out as a NEW frozen secondary set (`heidata_holdout`) that is never trained on
and never tuned on. The original frozen sets (`heidata_printed`, 7 books) stay
untouched — they remain the pre-registered adopt-if gate.

Usage:
    python scripts/build_heidata_training_sets.py
    python scripts/build_heidata_training_sets.py --freeze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "evals", "harness"))

import doc_data  # noqa: E402

TRAIN_BOOKS = ["giridharadasa1902", "nivajakavi1895", "sriramastavarajah1898",
               "trisuli1890", "vyasa1896", "vyasa1906"]
HOLDOUT_BOOKS = ["saktidharasukla1930", "simha1914"]


def build(zips_dir: str, out_root: str) -> dict:
    record = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "source": "heiDATA doi:10.11588/data/EGOKEI (CC BY 4.0)",
              "train_books": TRAIN_BOOKS, "holdout_books": HOLDOUT_BOOKS,
              "sets": {}}
    for name, books in (("heidata_train", TRAIN_BOOKS),
                        ("heidata_holdout", HOLDOUT_BOOKS)):
        out_dir = os.path.join(out_root, name)
        m = doc_data.build_heidata_dataset(zips_dir, out_dir, books=books)
        pages = len(m["entries"])
        lines = sum(e["n_lines"] for e in m["entries"])
        record["sets"][name] = {"pages": pages, "lines": lines,
                                "books": sorted({e["book"] for e in m["entries"]})}
        print(f"{name}: {pages} pages, {lines} lines "
              f"({len(record['sets'][name]['books'])} books)")
    with open(os.path.join(out_root, "heidata_split.json"), "w",
              encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.join(out_root, 'heidata_split.json')}")
    return record


def main():
    ap = argparse.ArgumentParser(description="Build heiDATA train/holdout split")
    ap.add_argument("--zips", default=os.path.join(BASE_DIR, "data", "doc_eval",
                                                   "heidata", "zips"))
    ap.add_argument("--out-root", default=os.path.join(BASE_DIR, "data",
                                                       "doc_eval"))
    ap.add_argument("--freeze", action="store_true",
                    help="freeze the holdout manifest (content hashes)")
    args = ap.parse_args()
    build(args.zips, args.out_root)
    if args.freeze:
        from eval_freeze import build_manifest
        out = os.path.join(BASE_DIR, "evals", "manifests",
                           "heidata_holdout_v1.json")
        build_manifest(os.path.join(args.out_root, "heidata_holdout"), "holdout",
                       out, license="CC BY 4.0",
                       provenance="heiDATA doi:10.11588/data/EGOKEI; two books "
                                  "held out from W1 attempt 3 training")
        print(f"frozen: {out}")


if __name__ == "__main__":
    main()
