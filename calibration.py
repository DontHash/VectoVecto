"""
calibration.py — confidence calibration maps (loaded by document_ocr).

The fitted file lives in calibration/ and is produced by fit_calibration.py on
a *development* set (never a frozen eval set). Only monotone (isotonic) maps
are shipped: temperature scaling was measured to be structurally unsuitable
for the Devanagari letterpress domain (saturated probabilities; BCE optimum
inverted the ranking, T<0).

No heavy imports; safe for the shipped path.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.join(BASE_DIR, "calibration",
                                   "rapidocr_devanagari_v1.json")

_CACHE: Dict[str, Dict] = {}


def load_calibration(path: Optional[str] = None) -> Optional[Dict]:
    """Load and cache a calibration JSON; None if missing or malformed."""
    path = path or DEFAULT_CALIBRATION
    key = os.path.abspath(path)
    if key in _CACHE:
        return _CACHE[key]
    try:
        with open(key, encoding="utf-8") as f:
            data = json.load(f)
        if "isotonic" not in data:
            return None
    except Exception:  # noqa: BLE001
        return None
    _CACHE[key] = data
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
