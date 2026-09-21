"""
eval_freeze.py — frozen eval manifests: content hashes + provenance.

Why: the pre-P4 evidence was scored on sets whose thresholds were tuned on the
same pages. A frozen manifest pins a dataset to exact file hashes, records its
provenance/license, and lets scoring refuse to run if any file drifted. The
contract (documented in the plan):

  * thresholds and heuristics are never tuned on a frozen set;
  * any change to a frozen dataset is a new freeze version (vN+1), and old
    numbers stay attached to the old version;
  * manifests are text-only and committed under evals/manifests/; the data
    itself stays in gitignored data/.

CLI:
    python eval_freeze.py --data-dir data/doc_eval/nepali_lines \
        --name nepali_lines_v1 --out evals/manifests/nepali_lines_v1.json \
        --license unknown --provenance "huggingface:gauravgiri/nepali-ocr-dataset"
    python eval_freeze.py --check evals/manifests/nepali_lines_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FREEZE_VERSION = 1


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _entry_files(data_dir: str, entry: Dict) -> Dict[str, str]:
    """Map logical role -> absolute path for one manifest entry."""
    out = {}
    for role, key in (("clean", "clean"), ("degraded", "degraded"), ("gt", "gt"),
                      ("image", "image")):
        rel = entry.get(key)
        if rel:
            out[role] = os.path.join(data_dir, rel)
    return out


def build_manifest(data_dir: str, name: str, out_path: str,
                   license: str = "unknown", provenance: str = "",
                   notes: str = "", version: int = FREEZE_VERSION,
                   entries: Optional[List[Dict]] = None) -> Dict:
    """Hash every entry file and write the freeze manifest."""
    if entries is None:
        with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
            entries = json.load(f)["entries"]
    frozen_entries = []
    for e in entries:
        rec = {"id": e.get("id")}
        for role, path in _entry_files(data_dir, e).items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"entry {e.get('id')}: missing {role} {path}")
            rec[role] = {
                "path": os.path.relpath(path, data_dir).replace("\\", "/"),
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
        frozen_entries.append(rec)
    manifest = {
        "freeze_version": version,
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "license": license,
        "provenance": provenance,
        "notes": notes,
        "n_entries": len(frozen_entries),
        "entries": frozen_entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_manifest(manifest_path: str, data_dir: Optional[str] = None,
                    expect_hash: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Check that the dataset on disk matches the frozen manifest.

    `data_dir` defaults to the manifest's `data_dir` field if present, else
    the manifest directory. Returns (ok, problems); problems lists every file
    that is missing, resized, or hash-changed.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if expect_hash and sha256_file(manifest_path) != expect_hash:
        return False, [f"manifest itself changed: {manifest_path}"]
    base = data_dir or manifest.get("data_dir") or os.path.dirname(manifest_path)
    problems: List[str] = []
    for e in manifest.get("entries", []):
        for role in ("clean", "degraded", "gt", "image"):
            rec = e.get(role)
            if not rec:
                continue
            path = os.path.join(base, rec["path"])
            if not os.path.exists(path):
                problems.append(f"{e.get('id')}:{role} missing ({rec['path']})")
                continue
            if rec.get("bytes") is not None and os.path.getsize(path) != rec["bytes"]:
                problems.append(f"{e.get('id')}:{role} size drift ({rec['path']})")
                continue
            if sha256_file(path) != rec["sha256"]:
                problems.append(f"{e.get('id')}:{role} hash drift ({rec['path']})")
    return (not problems), problems


def freeze_info(manifest_path: str) -> Dict:
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    return {"name": m.get("name"), "license": m.get("license"),
            "provenance": m.get("provenance"), "n_entries": m.get("n_entries"),
            "freeze_version": m.get("freeze_version"),
            "manifest_sha256": sha256_file(manifest_path)}


def main():
    ap = argparse.ArgumentParser(description="Freeze/verify an eval dataset")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--provenance", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--version", type=int, default=FREEZE_VERSION)
    ap.add_argument("--check", default=None, help="verify an existing manifest")
    args = ap.parse_args()

    if args.check:
        ok, problems = verify_manifest(args.check)
        info = freeze_info(args.check)
        print(f"[freeze] check {info['name']} ({info['manifest_sha256'][:12]}): "
              f"{'OK' if ok else 'DRIFT'}")
        for p in problems:
            print("  -", p)
        raise SystemExit(0 if ok else 1)

    if not (args.data_dir and args.name and args.out):
        raise SystemExit("provide --data-dir/--name/--out, or --check")
    m = build_manifest(args.data_dir, args.name, args.out, license=args.license,
                       provenance=args.provenance, notes=args.notes,
                       version=args.version)
    print(f"[freeze] {m['name']}: {m['n_entries']} entries -> {args.out}")


if __name__ == "__main__":
    main()
