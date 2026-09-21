"""
test_eval_freeze.py — frozen manifests and bootstrap CIs.

The freeze contract: a manifest pins every file's size + sha256; scoring must
refuse on any drift. The bootstrap CI must be deterministic for a given seed.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_metrics import bootstrap_ci  # noqa: E402
from eval_freeze import build_manifest, freeze_info, verify_manifest  # noqa: E402


def _make_dataset(tmp_path):
    pages = tmp_path / "pages"
    gt = tmp_path / "gt"
    pages.mkdir()
    gt.mkdir()
    entries = []
    for i in range(3):
        clean = pages / f"p{i}_clean.png"
        deg = pages / f"p{i}_degraded.png"
        gtf = gt / f"p{i}.txt"
        clean.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 32)
        deg.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i + 9]) * 32)
        gtf.write_text(f"पृष्ठ {i}", encoding="utf-8")
        entries.append({"id": f"p{i}", "clean": f"pages/p{i}_clean.png",
                        "degraded": f"pages/p{i}_degraded.png",
                        "gt": f"gt/p{i}.txt"})
    manifest = {"kind": "test", "entries": entries}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return entries


def test_freeze_build_and_verify(tmp_path):
    _make_dataset(tmp_path)
    out = tmp_path / "FROZEN.json"
    m = build_manifest(str(tmp_path), "unit_test_v1", str(out),
                       license="CC-BY-4.0", provenance="unit-test")
    assert m["n_entries"] == 3
    assert m["license"] == "CC-BY-4.0"
    for e in m["entries"]:
        assert len(e["clean"]["sha256"]) == 64
        assert e["gt"]["path"].startswith("gt/")

    ok, problems = verify_manifest(str(out), data_dir=str(tmp_path))
    assert ok and problems == []
    info = freeze_info(str(out))
    assert info["n_entries"] == 3 and info["freeze_version"] == 1


def test_freeze_detects_hash_drift(tmp_path):
    _make_dataset(tmp_path)
    out = tmp_path / "FROZEN.json"
    build_manifest(str(tmp_path), "unit_test_v1", str(out))
    # tamper one byte in one page
    p = tmp_path / "pages" / "p1_clean.png"
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(bytes(data))
    ok, problems = verify_manifest(str(out), data_dir=str(tmp_path))
    assert not ok
    assert any("p1:clean" in x and "hash drift" in x for x in problems)


def test_freeze_detects_missing_file(tmp_path):
    _make_dataset(tmp_path)
    out = tmp_path / "FROZEN.json"
    build_manifest(str(tmp_path), "unit_test_v1", str(out))
    os.remove(tmp_path / "gt" / "p2.txt")
    ok, problems = verify_manifest(str(out), data_dir=str(tmp_path))
    assert not ok
    assert any("p2:gt" in x and "missing" in x for x in problems)


def test_bootstrap_ci_deterministic_and_bounded():
    vals = [0.10, 0.20, 0.30, 0.40, 0.50]
    a = bootstrap_ci(vals, n_boot=500, seed=7)
    b = bootstrap_ci(vals, n_boot=500, seed=7)
    assert a == b
    assert a[0] <= 0.30 <= a[1]
    assert bootstrap_ci([0.25]) == [0.25, 0.25]
    lo, hi = bootstrap_ci([0.4, 0.4, 0.4])
    assert lo == hi == pytest.approx(0.4)
