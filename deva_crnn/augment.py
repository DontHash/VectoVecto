"""
augment.py — print-degradation augmentation for line crops (local use).

Kept out of `data.py` so the training path never imports OpenCV: the Vertex
PyTorch container has torch/PIL/numpy but not cv2.

`level="light"` is the attempt-1 augmentation (blur/noise/JPEG/brightness).
`level="heavy"` adds the artifacts that actually separate a rendered line from
a scanned one: ink spread/erosion, resolution loss (downscale-upscale), uneven
illumination, stronger blur/noise/JPEG and a slight rotation.
"""
from __future__ import annotations

import cv2
import numpy as np


def augment_line(img_bgr: np.ndarray, rng: np.random.Generator,
                 level: str = "light") -> np.ndarray:
    """Deterministic print-degradation augmentation (given `rng`)."""
    if level == "heavy":
        return _heavy(img_bgr, rng)
    return _light(img_bgr, rng)


def _light(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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


def _heavy(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    h, w = out.shape[:2]

    # Ink spread (heavy inking / letterpress bleed) or erosion (thin print).
    if rng.random() < 0.55:
        k = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5:
            out = cv2.dilate(out, k, iterations=1).astype(np.float32)
        else:
            out = cv2.erode(out, k, iterations=1).astype(np.float32)

    # Resolution loss: a 300-dpi scan of small print is not crisp.
    if rng.random() < 0.5:
        f = float(rng.uniform(0.45, 0.8))
        small = cv2.resize(out, (max(4, int(w * f)), max(4, int(h * f))),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Anisotropic squeeze/stretch: detector boxes are tight around cells and
    # long lines alike, so the recognizer must not assume a fixed aspect.
    if rng.random() < 0.4:
        fx = float(rng.uniform(0.65, 1.45))
        out = cv2.resize(out, (max(4, int(w * fx)), h),
                         interpolation=cv2.INTER_LINEAR)
        h, w = out.shape[:2]

    # Uneven illumination (page curvature / phone shadow).
    if rng.random() < 0.4:
        ramp = np.linspace(float(rng.uniform(0.72, 0.88)),
                           float(rng.uniform(1.0, 1.12)), w,
                           dtype=np.float32)
        out = out * ramp[None, :, None]

    if rng.random() < 0.8:
        sigma = float(rng.uniform(0.5, 2.2))
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    if rng.random() < 0.7:
        out = out + rng.normal(0.0, float(rng.uniform(4, 20)), out.shape)
    out = np.clip(out * float(rng.uniform(0.7, 1.2)) +
                  float(rng.uniform(-20, 20)), 0, 255)

    if rng.random() < 0.7:
        q = int(rng.integers(28, 72))
        ok, enc = cv2.imencode(".jpg", out.astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

    if rng.random() < 0.5:
        ang = float(rng.uniform(-1.5, 1.5))
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 255).astype(np.uint8)
