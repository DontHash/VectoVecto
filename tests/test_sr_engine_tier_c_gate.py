"""
test_sr_engine_tier_c_gate.py — the rejected tier_c photo checkpoint must never
be picked by 'auto' unless artifacts/tier_c/ACCEPTED exists (gcp/RESULTS.md).
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import sr_engine  # noqa: E402


def test_tier_c_ignored_without_accepted_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(sr_engine, "TIER_C_ACCEPTED_MARKER",
                        str(tmp_path / "ACCEPTED"))
    assert sr_engine.tier_c_accepted() is False
    assert sr_engine._tier_c_candidates() == []


def test_tier_c_eligible_with_accepted_marker(tmp_path, monkeypatch):
    marker = tmp_path / "ACCEPTED"
    marker.write_text("accepted\n", encoding="utf-8")
    monkeypatch.setattr(sr_engine, "TIER_C_ACCEPTED_MARKER", str(marker))
    assert sr_engine.tier_c_accepted() is True
    candidates = sr_engine._tier_c_candidates()
    assert candidates, "accepted tier_c must expose candidates"
    assert candidates[0].endswith(os.path.join("tier_c", "g_ema.pth"))


def test_auto_prefers_x4plus_when_tier_c_not_accepted(tmp_path, monkeypatch):
    # Gate off: the real tier_c files on this machine (if any) must be skipped.
    monkeypatch.setattr(sr_engine, "TIER_C_ACCEPTED_MARKER",
                        str(tmp_path / "ACCEPTED"))
    assert sr_engine._tier_c_candidates() == []

    fake_x4plus = tmp_path / "RealESRGAN_x4plus.pth"
    fake_x4plus.write_bytes(b"stub")
    monkeypatch.setattr(sr_engine, "_x4plus_candidates", lambda: [str(fake_x4plus)])

    seen = {}

    class _Dummy:
        def __init__(self, path, *a, **k):
            seen["path"] = path

    monkeypatch.setattr(sr_engine, "TorchEngine", _Dummy)
    sr_engine.load_engine("auto", device="cpu")
    assert seen.get("path") == str(fake_x4plus), \
        "auto must fall through to x4plus when tier_c is not accepted"
