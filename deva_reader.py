"""
deva_reader.py — optional Devanagari line reader (the W1 recognizer).

The recognizer is trained by `deva_crnn/` and is adopted only when it passes
the pre-registered gate: frozen heiDATA digit-exact 0.810 [0.769, 0.847] at
h=48/W=512 (docs/PLAN.md Appendix R5). It replaces the *recognition* of
Devanagari line boxes while RapidOCR keeps doing detection; every line falls
back to the backend reading when the model output is empty or implausible.

The weights are deliberately **not** part of this repository. Deployments
provide them via `VECTOVECTO_DEVA_CKPT` or `weights/deva_crnn_h48w512.pt`
(both git-ignored); without them the reader is simply inactive.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_VAR = "VECTOVECTO_DEVA_CKPT"
DEFAULT_CKPT = os.path.join(BASE_DIR, "weights", "deva_crnn_h48w512.pt")
MIN_CHARS = 1
MAX_CHARS = 64


def resolve_ckpt(explicit: Optional[str] = None) -> Optional[str]:
    """First existing checkpoint: explicit path, env var, repo default."""
    for cand in (explicit, os.environ.get(ENV_VAR), DEFAULT_CKPT):
        if cand and os.path.exists(cand):
            return cand
    return None


def _has_vertical_rule(img: Optional[np.ndarray], x0: int, x1: int,
                       y0: int, y1: int, dark: int = 128,
                       min_frac: float = 0.6) -> bool:
    """True when a drawn vertical rule separates two boxes (table columns).

    Adjacent table cells are geometrically indistinguishable from line
    fragments (measured on born-digital court registers); the drawn column
    rule is the difference, so a dark vertical stroke in the gap blocks the
    merge.
    """
    if img is None or x1 - x0 < 1 or y1 - y0 < 2:
        return False
    band = img[y0:y1, x0:x1]
    if band.size == 0:
        return False
    gray = band if band.ndim == 2 else band.mean(axis=2)
    col_dark = (gray < dark).mean(axis=0)
    return bool((col_dark > min_frac).any())


def merge_line_boxes(tokens, img: Optional[np.ndarray] = None,
                     max_gap_ratio: float = 0.75,
                     min_overlap: float = 0.5):
    """Group adjacent token boxes into text lines (fragments -> one line).

    RapidOCR's detector sometimes splits a printed line into 2-3 fragments
    (measured on letterpress pages: 33 tokens for 21 ALTO lines). The line
    recognizer was trained on whole lines, so fragments make it invent line
    endings; merging them first fixes the granularity mismatch. Isolated
    boxes (table cells) stay single, columns stay separate because their
    horizontal gap exceeds the threshold, and a drawn vertical rule in the
    gap blocks the merge outright.

    Returns [(bbox, [token indices])] in reading order.
    """
    if not tokens:
        return []
    order = sorted(range(len(tokens)),
                   key=lambda i: (tokens[i].bbox[1], tokens[i].bbox[0]))
    groups = []
    for i in order:
        x0, y0, x1, y1 = tokens[i].bbox
        h = max(1, y1 - y0)
        placed = False
        for g in groups:
            gx0, gy0, gx1, gy1 = g["bbox"]
            gh = max(1, gy1 - gy0)
            overlap = min(gy1, y1) - max(gy0, y0)
            if overlap < min_overlap * min(h, gh):
                continue
            gap = x0 - gx1
            if gap < 0:  # overlapping x-ranges: same line
                gap = 0
            if gap > max_gap_ratio * min(h, gh):
                continue
            if gap > 0 and _has_vertical_rule(img, gx1, x0,
                                              max(gy0, y0), min(gy1, y1)):
                continue  # a table rule: separate cells, never one line
            g["bbox"] = (min(gx0, x0), min(gy0, y0),
                         max(gx1, x1), max(gy1, y1))
            g["idx"].append(i)
            placed = True
            break
        if not placed:
            groups.append({"bbox": (x0, y0, x1, y1), "idx": [i]})
    groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
    return [(g["bbox"], g["idx"]) for g in groups]


def page_is_line_like(tokens, page_width: int, min_median_aspect: float = 5.0,
                      min_width_frac: float = 0.25) -> bool:
    """True when the detector produced line-shaped boxes (running text).

    The recognizer is a *line* model: it was trained on whole printed lines and
    measured to help there (letterpress books: -42% page CER) and to hurt where
    the detector yields table cells (born-digital court registers, degraded
    photos of them). Rather than classify styles, `auto` asks the geometry:
    running-text pages have wide, line-shaped boxes; table pages do not.
    """
    if not tokens:
        return False
    aspects, widths = [], []
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        aspects.append(max(1, x1 - x0) / max(1, y1 - y0))
        widths.append((x1 - x0) / max(1, page_width))
    return (float(np.median(aspects)) >= min_median_aspect
            and float(np.median(widths)) >= min_width_frac)


class DevaLineReader:
    """Batched line recognition with the trained CRNN+CTC model."""

    def __init__(self, ckpt_path: str):
        from deva_crnn.predict import load_model
        self.ckpt_path = ckpt_path
        self.model, self.charset = load_model(ckpt_path, "cpu")
        self.in_h = int(getattr(self.model, "in_h", 32))
        self.in_w = int(getattr(self.model, "in_w", 256))

    def read(self, crops: List[np.ndarray]) -> List[str]:
        from deva_crnn.predict import recognize_lines
        return recognize_lines(self.model, self.charset, crops)

    @staticmethod
    def plausible(text: str) -> bool:
        """Guard against implausible model output (keep the backend reading)."""
        text = (text or "").strip()
        return MIN_CHARS <= len(text) <= MAX_CHARS


_CACHE: Dict[str, DevaLineReader] = {}
_LOCK = threading.Lock()


def get_reader(ckpt: Optional[str] = None) -> Optional[DevaLineReader]:
    """Cached reader, or None when no checkpoint is available."""
    path = resolve_ckpt(ckpt)
    if not path:
        return None
    with _LOCK:
        if path not in _CACHE:
            try:
                _CACHE[path] = DevaLineReader(path)
            except Exception:  # noqa: BLE001 - never break OCR over the reader
                return None
        return _CACHE[path]


def reader_info(ckpt: Optional[str] = None) -> Dict:
    """Small status dict for CLI/JSON output and the deployment docs."""
    path = resolve_ckpt(ckpt)
    if not path:
        return {"active": False, "reason": "no checkpoint",
                "env": ENV_VAR, "default": DEFAULT_CKPT}
    reader = get_reader(path)
    if reader is None:
        return {"active": False, "reason": "checkpoint failed to load",
                "ckpt": path}
    return {"active": True, "ckpt": os.path.basename(path),
            "in_h": reader.in_h, "in_w": reader.in_w,
            "charset": len(reader.charset)}
