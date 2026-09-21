"""
data.py — line-image dataset for the CRNN: npz export + torch Dataset.

The npz format (`images` uint8 NxHxW, `texts` unicode array) is compact enough
to ship inside the Vertex AI python package and fast to load.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

IN_H = 32
IN_W = 256


def normalize_line(img_bgr: np.ndarray, h: int = IN_H, w: int = IN_W) -> np.ndarray:
    """Grayscale, height-normalized, right-padded line image (uint8)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    scale = h / gray.shape[0]
    new_w = min(w, max(4, int(round(gray.shape[1] * scale))))
    resized = cv2.resize(gray, (new_w, h), interpolation=cv2.INTER_AREA)
    out = np.full((h, w), 255, dtype=np.uint8)
    out[:, :new_w] = resized
    return out


def export_npz(images: List[np.ndarray], texts: List[str], path: str,
               h: int = IN_H, w: int = IN_W) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.stack([normalize_line(im, h=h, w=w) for im in images])
    np.savez_compressed(path, images=arr,
                        texts=np.array(texts, dtype=object).astype(str))
    return path


def load_npz(path: str) -> Tuple[np.ndarray, List[str]]:
    d = np.load(path, allow_pickle=False)
    return d["images"], [str(t) for t in d["texts"]]
