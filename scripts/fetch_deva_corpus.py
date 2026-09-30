"""
fetch_deva_corpus.py — CC-100 Nepali sentence pool for the W1 synthetic mix (W-C).

The CRNN's synthetic lines are drawn from a small hand-built word pool
(`doc_data._DEVA_LINE_WORDS`, ~66 words). W-C tests whether real sentence
structure improves the frozen line gate, so this script samples a sentence
pool from CC-100 Nepali (`ne.txt.xz`, 393 MB, statmt.org).

License/provenance: CC-100 states "no claims of intellectual property are made
on the work of preparation of the corpus" (CCNet over January-December 2018
Common Crawl snapshots; cite Conneau et al. 2020 / Wenzek et al. 2020). The
output is not redistributed (`data/` is gitignored); the provenance row is in
docs/LICENSES.md.

Filters per line (streaming, deterministic given the source file):
  * 10-60 chars after whitespace normalization
  * >= 60% Devanagari among letters
  * no replacement chars, URLs or ASCII-letter-heavy lines
  * no invalid Devanagari combining sequences (`doc_metrics.devanagari_validity`)

Usage:
    python scripts/fetch_deva_corpus.py
    python scripts/fetch_deva_corpus.py --limit 100000 --local ne.txt.xz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from typing import Dict, Iterator, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.core.metrics import devanagari_validity, normalize_text  # noqa: E402

CORPUS_URL = "https://data.statmt.org/cc-100/ne.txt.xz"
CORPUS_NAME = "deva_sentences_v1.txt"
MANIFEST_NAME = "deva_sentences_v1.manifest.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) VeriScript-Research/1.0 "
      "(offline corpus build; contact: local)")
MIN_CHARS = 10
MAX_CHARS = 60
MIN_DEVA_SHARE = 0.6
# Charset control (W-C): only Devanagari, ASCII digits, whitespace and common
# punctuation enter the training pool, so the corpus run differs from the v8
# pool in *text structure*, not in output alphabet size.
_PUNCT_ALLOWED = set(".,?!:;'\"()[]-/%")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _deva_char_ok(c: str) -> bool:
    return ("\u0900" <= c <= "\u097f") or c.isdigit() or c.isspace() \
        or c in _PUNCT_ALLOWED


def _download(url: str, dst: str, retries: int = 3) -> None:
    """Atomic download (..part -> replace) with retries; no partials left."""
    part = dst + ".part"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as resp, \
                    open(part, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
            os.replace(part, dst)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    try:
        os.remove(part)
    except OSError:
        pass
    raise RuntimeError(f"download failed after {retries}: {last_err}")


def iter_corpus_lines(path: str) -> Iterator[str]:
    """Stream decompressed lines from the .xz file (no full load)."""
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def line_ok(line: str, min_chars: int = MIN_CHARS,
            max_chars: int = MAX_CHARS) -> bool:
    """One sentence: length, script mix, charset, no damage, no URLs."""
    text = normalize_text(line)
    if not (min_chars <= len(text) <= max_chars):
        return False
    if "\ufffd" in text or _URL_RE.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    deva = sum(1 for c in letters if "\u0900" <= c <= "\u097f")
    if deva / len(letters) < MIN_DEVA_SHARE:
        return False
    if not all(_deva_char_ok(c) for c in text):
        return False
    return devanagari_validity(text)["invalid_tokens"] == 0


def sample_corpus(lines: Iterator[str], limit: int, seed: int) -> List[str]:
    """Deterministic reservoir sample over an already-filtered stream."""
    rng = random.Random(seed)
    out: List[str] = []
    for i, line in enumerate(lines):
        if len(out) < limit:
            out.append(line)
        else:
            j = rng.randint(0, i)
            if j < limit:
                out[j] = line
    return out


def build(local_path: str, limit: int = 100000, seed: int = 1,
          min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> Dict:
    """Filter + sample in one streaming pass; returns lines + stats."""
    rng = random.Random(seed)
    sampled: List[str] = []
    seen = kept = 0
    for raw in iter_corpus_lines(local_path):
        seen += 1
        if not line_ok(raw, min_chars, max_chars):
            continue
        line = normalize_text(raw)
        kept += 1
        if len(sampled) < limit:
            sampled.append(line)
        else:
            j = rng.randint(0, kept - 1)
            if j < limit:
                sampled[j] = line
    return {
        "lines": sampled,
        "stats": {"source_lines": seen, "kept": kept,
                  "sampled": len(sampled), "limit": limit, "seed": seed,
                  "min_chars": min_chars, "max_chars": max_chars},
    }


def write_corpus(lines: List[str], out_path: str) -> str:
    """One sentence per line (LF); returns sha256 of the file bytes."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    text = "".join(line + "\n" for line in lines)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch(out_dir: str, local: Optional[str] = None, limit: int = 100000,
          seed: int = 1) -> Dict:
    """Download (or reuse local), filter, sample, write corpus + manifest."""
    os.makedirs(out_dir, exist_ok=True)
    src = local or os.path.join(out_dir, os.path.basename(CORPUS_URL))
    if not local and not os.path.exists(src):
        _download(CORPUS_URL, src)
    raw_sha = hashlib.sha256()
    with open(src, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            raw_sha.update(block)
    result = build(src, limit=limit, seed=seed)
    out_path = os.path.join(out_dir, CORPUS_NAME)
    out_sha = write_corpus(result["lines"], out_path)
    manifest = {
        "kind": "deva_corpus",
        "version": 1,
        "name": "deva_sentences_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "url": CORPUS_URL,
            "file": os.path.basename(src),
            "bytes": os.path.getsize(src),
            "sha256": raw_sha.hexdigest(),
            "license": ("CC-100 (statmt.org): no claims of intellectual "
                        "property on the preparation; 2018 Common Crawl; "
                        "cite Conneau et al. 2020 / Wenzek et al. 2020"),
        },
        "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                    "min_deva_share": MIN_DEVA_SHARE,
                    "invalid_sequences": 0, "no_urls": True,
                    "charset": "devanagari + ascii digits + . , ? ! : ; ' \" "
                               "( ) [ ] - / %"},
        "stats": result["stats"],
        "output": {"file": CORPUS_NAME, "sha256": out_sha,
                   "bytes": os.path.getsize(out_path),
                   "lines": len(result["lines"])},
    }
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description="Build the CC-100 Nepali sentence pool (W-C)")
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data",
                                                      "corpus"))
    ap.add_argument("--local", default=None,
                    help="local ne.txt.xz copy (air-gapped builds)")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    manifest = fetch(args.out_dir, args.local, args.limit, args.seed)
    s, out = manifest["stats"], manifest["output"]
    print(f"[corpus] {s['kept']} lines passed the filters "
          f"({s['source_lines']} seen); sampled {out['lines']}")
    print(f"[corpus] wrote {os.path.join(args.out_dir, out['file'])} "
          f"sha256 {out['sha256']}")


if __name__ == "__main__":
    main()
