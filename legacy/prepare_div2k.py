"""
prepare_div2k.py — fetch and cache the DIV2K training set.

DIV2K is the standard SR training dataset: 800 HR train images + 100 HR
validation images, 2K resolution, ~3GB. This helper downloads + unpacks it
into ./data/DIV2K/ so train_deep_sr.py can consume it directly.

If you have a Kaggle kernel (Tier B target), DIV2K is already available as
the Kaggle dataset "div2k-dataset" — you can attach it instead of running
this script. This script is for local/laptop use.

Usage:
    python prepare_div2k.py          # download train + val
    python prepare_div2k.py --val    # only validation
"""

# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# archived research modules live under legacy/research; harness under evals/harness
for _extra in (_os.path.join(_ROOT, "legacy", "research"),
               _os.path.join(_ROOT, "evals", "harness")):
    if _extra not in _sys.path:
        _sys.path.insert(0, _extra)

import argparse
import os
import sys
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "DIV2K")

# Official DIV2K URLs (data.vision.ee.ethz.ch)
DIV2K_URLS = {
    "train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "train_LR_bicubic_X4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "valid_LR_bicubic_X4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
}


def _download(url, dest):
    print(f"Downloading {url}")
    if os.path.exists(dest):
        print(f"  already exists -> skip: {dest}")
        return
    urllib.request.urlretrieve(url, dest)
    print(f"  saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def _unzip(zip_path, extract_to):
    print(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def prepare(val_only=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    keys = ["valid_HR", "valid_LR_bicubic_X4"] if val_only else list(DIV2K_URLS.keys())
    for k in keys:
        url = DIV2K_URLS[k]
        zp = os.path.join(DATA_DIR, os.path.basename(url))
        _download(url, zp)
        _unzip(zp, DATA_DIR)
    print(f"\nDIV2K ready in: {DATA_DIR}")
    print("Contents:")
    for root, dirs, files in os.walk(DATA_DIR):
        depth = root[len(DATA_DIR):].count(os.sep)
        if depth <= 1 and (root.endswith("HR") or root.endswith("LR_bicubic") or
                            root.endswith("X4")):
            print(f"  {os.path.basename(root)}: {len(files)} images")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val", action="store_true", help="Only download validation set")
    args = p.parse_args()
    prepare(val_only=args.val)