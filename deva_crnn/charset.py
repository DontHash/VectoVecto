"""
charset.py — CTC charset for Devanagari line recognition.

Characters come from the training labels; the blank is index 0 (CTC). Encoding
is per-codepoint (no shaping - the image carries the shaping).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

BLANK = 0


def build_charset(texts: Iterable[str]) -> List[str]:
    chars = sorted({c for t in texts for c in t})
    return ["\u0000"] + chars


def encode(text: str, charset: List[str]) -> List[int]:
    idx = {c: i for i, c in enumerate(charset)}
    return [idx[c] for c in text if c in idx]


def decode(ids: Iterable[int], charset: List[str]) -> str:
    out: List[str] = []
    prev = None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(charset[i])
        prev = i
    return "".join(out)
