"""
test_build_deva_wheel.py — Vertex wheel staging (no pip build).
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import build_deva_wheel as bw  # noqa: E402


def test_stage_package_copies_sources_npz_and_setup(tmp_path):
    src = tmp_path / "deva_crnn"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "train.py").write_text("x = 1\n", encoding="utf-8")
    (src / "data").mkdir()
    npz = tmp_path / "train.npz"
    npz.write_bytes(b"PK\x03\x04fake")
    stage = tmp_path / "stage"
    bw.stage_package(str(src), str(npz), str(stage), "9.9.9")
    assert (stage / "deva_crnn" / "train.py").exists()
    assert (stage / "deva_crnn" / "__init__.py").exists()
    assert (stage / "deva_crnn" / "data" / "train.npz").read_bytes() == \
        b"PK\x03\x04fake"
    setup = (stage / "setup.py").read_text(encoding="utf-8")
    assert 'version="9.9.9"' in setup and "data/*.npz" in setup


def test_stage_package_replaces_existing_stage(tmp_path):
    src = tmp_path / "deva_crnn"
    src.mkdir()
    (src / "a.py").write_text("", encoding="utf-8")
    npz = tmp_path / "d.npz"
    npz.write_bytes(b"x")
    stage = tmp_path / "stage"
    (stage / "deva_crnn").mkdir(parents=True)
    (stage / "stale.txt").write_text("old", encoding="utf-8")
    bw.stage_package(str(src), str(npz), str(stage), "0.0.1")
    assert not (stage / "stale.txt").exists()
