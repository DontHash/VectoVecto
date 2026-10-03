"""
calibration.py — confidence calibration maps (loaded by document_ocr).

The fitted file lives in `calibration/` (source checkout) or the wheel
data-files location `<sys.prefix>/calibration/`; it is produced by
fit_calibration.py on a *development* set (never a frozen eval set). Only
monotone (isotonic) maps are shipped: temperature scaling was measured to be
structurally unsuitable for the Devanagari letterpress domain (saturated
probabilities; BCE optimum inverted the ranking, T<0).

No heavy imports; safe for the shipped path.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

from veriscript.paths import ROOT

BASE_DIR = ROOT
CALIBRATION_NAME = "rapidocr_devanagari_v1.json"
DEFAULT_CALIBRATION = os.path.join(BASE_DIR, "calibration", CALIBRATION_NAME)

_CACHE: Dict[str, Dict] = {}


def resolve_calibration(path: Optional[str] = None) -> Optional[str]:
    """First existing calibration file, or None.

    `path` (when given) is exact; otherwise the repo copy and the wheel
    data-files location are probed in order.
    """
    if path:
        return os.path.abspath(path) if os.path.isfile(path) else None
    for cand in (DEFAULT_CALIBRATION,
                 os.path.join(sys.prefix, "calibration", CALIBRATION_NAME)):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def load_calibration(path: Optional[str] = None) -> Optional[Dict]:
    """Load and cache a calibration JSON; None if missing or malformed."""
    resolved = resolve_calibration(path)
    if resolved is None:
        return None
    if resolved in _CACHE:
        return _CACHE[resolved]
    try:
        with open(resolved, encoding="utf-8") as f:
            data = json.load(f)
        if "isotonic" not in data:
            return None
    except Exception:  # noqa: BLE001
        return None
    _CACHE[resolved] = data
    return data


def apply_isotonic(conf_pct: float, mapping: Dict) -> float:
    """Map a 0-100 confidence through the piecewise-constant isotonic table."""
    xs, ys = mapping.get("x", []), mapping.get("y", [])
    if not xs or not ys:
        return conf_pct
    x = conf_pct / 100.0
    y = ys[0]
    for xi, yi in zip(xs, ys):
        if x >= xi:
            y = yi
        else:
            break
    return max(0.0, min(100.0, y * 100.0))
