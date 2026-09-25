"""
build_deva_wheel.py — package deva_crnn + a training npz as a Vertex AI wheel (W-C).

The Vertex custom job installs the trainer from a wheel and the npz travels
inside it (`deva_crnn.train.resolve_data_path` resolves `pkg://data/...`) —
the same pattern as the attempt-1 job. Staging + `pip wheel` lives here so the
recipe is reproducible and the wheel version is explicit.

Usage:
    python scripts/build_deva_wheel.py \
        --npz deva_crnn/data/train_v9_corpus_h48w512.npz \
        --version 0.2.0 --out out/wheel_v9
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP_PY = """from setuptools import setup

setup(
    name="deva_crnn",
    version="{version}",
    packages=["deva_crnn"],
    package_data={{"deva_crnn": ["data/*.npz"]}},
    include_package_data=True,
    install_requires=["numpy", "torch"],
)
"""


def stage_package(src_dir: str, npz: str, stage_dir: str,
                  version: str) -> str:
    """Copy `deva_crnn/*.py` + the npz into a buildable stage dir."""
    pkg = os.path.join(stage_dir, "deva_crnn")
    data = os.path.join(pkg, "data")
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(data, exist_ok=True)
    for path in glob.glob(os.path.join(src_dir, "*.py")):
        shutil.copy(path, os.path.join(pkg, os.path.basename(path)))
    shutil.copy(npz, os.path.join(data, os.path.basename(npz)))
    with open(os.path.join(stage_dir, "setup.py"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(SETUP_PY.format(version=version))
    return pkg


def build_wheel(src_dir: str, npz: str, out_dir: str, version: str) -> str:
    """Stage and `pip wheel`; returns the wheel path."""
    stage = os.path.join(out_dir, "stage")
    dist = os.path.join(out_dir, "dist")
    os.makedirs(dist, exist_ok=True)
    stage_package(src_dir, npz, stage, version)
    cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps",
           "--no-build-isolation", "-w", dist, stage]
    print("[wheel]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    wheels = glob.glob(os.path.join(dist, "deva_crnn-*.whl"))
    if not wheels:
        raise SystemExit("pip wheel produced no deva_crnn wheel")
    wheel = max(wheels, key=os.path.getmtime)
    print(f"[wheel] {wheel} ({os.path.getsize(wheel) / 1e6:.1f} MB)")
    return wheel


def main():
    ap = argparse.ArgumentParser(description="Build the Vertex deva_crnn wheel")
    ap.add_argument("--src", default=os.path.join(BASE_DIR, "deva_crnn"))
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "out", "wheel_v9"))
    ap.add_argument("--version", default="0.2.0")
    args = ap.parse_args()
    build_wheel(args.src, args.npz, args.out, args.version)


if __name__ == "__main__":
    main()
