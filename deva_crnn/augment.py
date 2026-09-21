"""
augment.py — print-degradation augmentation for line crops (local use).

Kept out of `data.py` so the training path never imports OpenCV: the Vertex
PyTorch container has torch/PIL/numpy but not cv2.
"""
from __future__ import annotations

import cv2
import numpy as np


def augment_line(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap print-degradation augmentation (blur/noise/JPEG/brightness)."""
    out = img_bgr.astype(np.float32)
    if rng.random() < 0.5:
        sigma = float(rng.uniform(0.4, 1.3))
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    if rng.random() < 0.4:
        out = out + rng.normal(0.0, float(rng.uniform(3, 12)), out.shape)
    out = np.clip(out * float(rng.uniform(0.85, 1.15)) +
                  float(rng.uniform(-12, 12)), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        q = int(rng.integers(40, 85))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out
