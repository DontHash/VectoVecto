"""
degradation_document.py — deterministic document-page degradations for the
document eval harness.

Photo kernels live in `degradation_v2.py` (camera/lens/JPEG pipeline for SR
training). Document pages fail differently: uneven phone-flash illumination,
lost resolution from capture distance, motion blur from shaky hands, slight
perspective, paper speckle, and recompression. This module models those.

API:
    degrade_page(img_bgr, seed=42, level="medium") -> np.ndarray   # uint8 BGR in/out
    make_pair(...)  # clean + degraded (+ seed metadata)

All randomness is seeded: same (image, seed, level) -> same output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import numpy as np

Level = str  # "mild" | "medium" | "heavy"

PRESETS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    # step: (probability, lo, hi)
    "mild": {
        "rotate": (0.5, 0.3, 1.2),
        "perspective": (0.4, 0.005, 0.015),
        "defocus": (0.4, 0.8, 1.6),
        "motion": (0.3, 3, 9),
        "downscale": (0.5, 0.6, 0.75),
        "illumination": (0.6, 0.10, 0.28),
        "speckle": (0.4, 0.0005, 0.003),
        "jpeg": (0.8, 55, 85),
    },
    "medium": {
        "rotate": (0.6, 0.5, 2.0),
        "perspective": (0.5, 0.008, 0.025),
        "defocus": (0.5, 1.0, 2.0),
        "motion": (0.4, 4, 13),
        "downscale": (0.7, 0.45, 0.65),
        "illumination": (0.8, 0.18, 0.40),
        "speckle": (0.5, 0.001, 0.006),
        "jpeg": (0.9, 40, 70),
    },
    "heavy": {
        "rotate": (0.7, 1.0, 3.0),
        "perspective": (0.6, 0.012, 0.035),
        "defocus": (0.6, 1.4, 2.6),
        "motion": (0.5, 6, 18),
        "downscale": (0.8, 0.35, 0.55),
        "illumination": (0.9, 0.30, 0.55),
        "speckle": (0.6, 0.002, 0.012),
        "jpeg": (1.0, 30, 55),
    },
}


@dataclass
class DegradeMeta:
    applied: Dict[str, float] = field(default_factory=dict)
    seed: int = 0
    level: str = "medium"


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def _rotate(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    angle = float(rng.uniform(-hi, hi)) if rng.random() < 0.5 else float(rng.uniform(lo, hi))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, angle


def _perspective(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    h, w = img.shape[:2]
    amt = float(rng.uniform(lo, hi)) * max(h, w)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + rng.normal(0.0, amt, size=src.shape).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return out


def _blur_defocus(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    sigma = float(rng.uniform(lo, hi))
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def _blur_motion(img: np.ndarray, rng: np.random.Generator, lo: int, hi: int) -> np.ndarray:
    length = int(rng.integers(int(lo), int(hi) + 1))
    angle = float(rng.uniform(0, 180))
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    s = k.sum()
    if s > 0:
        k /= s
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)


def _downscale_up(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    """Simulate capture at lower resolution: shrink then restore page size."""
    h, w = img.shape[:2]
    f = float(rng.uniform(lo, hi))
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _illumination(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    """Smooth multiplicative lighting field + optional elliptical shadow."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = xx / max(w - 1, 1), yy / max(h - 1, 1)
    amp = float(rng.uniform(lo, hi))

    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(3):
        g = rng.normal(0, 1)
        field += g * np.sin(np.pi * (rng.uniform(0.5, 2.0) * gx + rng.uniform(0, 0.3))) * \
            np.sin(np.pi * (rng.uniform(0.5, 2.0) * gy + rng.uniform(0, 0.3)))
    field = field / (np.abs(field).max() + 1e-6)
    field *= amp

    if rng.random() < 0.5:
        cx, cy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        rx, ry = rng.uniform(0.25, 0.5), rng.uniform(0.25, 0.5)
        d = ((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2
        shadow = np.exp(-d * 2.0) * amp * 0.8
        field = field - shadow

    gain = 1.0 + field
    out = img.astype(np.float32) * gain[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _speckle(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    p = float(rng.uniform(lo, hi))
    n = int(h * w * p)
    if n > 0:
        ys = rng.integers(0, h, n)
        xs = rng.integers(0, w, n)
        vals = np.where(rng.random(n) < 0.5, 0, 255).astype(np.uint8)
        out[ys, xs] = vals[:, None]
    sigma = float(rng.uniform(1.0, 4.0))
    noise = rng.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    q = int(rng.integers(int(lo), int(hi) + 1))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def degrade_page(img_bgr: np.ndarray, seed: int = 42, level: Level = "medium") -> np.ndarray:
    """Apply a seeded document capture degradation. uint8 BGR in/out."""
    if level not in PRESETS:
        raise ValueError(f"level must be one of {sorted(PRESETS)}, got {level!r}")
    if img_bgr.dtype != np.uint8:
        img = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        img = img_bgr.copy()

    rng = np.random.default_rng(seed)
    p = PRESETS[level]
    out = img

    geom = []
    if rng.random() < p["perspective"][0]:
        out = _perspective(out, rng, p["perspective"][1], p["perspective"][2])
    if rng.random() < p["rotate"][0]:
        out, angle = _rotate(out, rng, p["rotate"][1], p["rotate"][2])
        geom.append(angle)

    if rng.random() < p["motion"][0]:
        out = _blur_motion(out, rng, p["motion"][1], p["motion"][2])
    if rng.random() < p["defocus"][0]:
        out = _blur_defocus(out, rng, p["defocus"][1], p["defocus"][2])

    if rng.random() < p["downscale"][0]:
        out = _downscale_up(out, rng, p["downscale"][1], p["downscale"][2])

    if rng.random() < p["illumination"][0]:
        out = _illumination(out, rng, p["illumination"][1], p["illumination"][2])

    if rng.random() < p["speckle"][0]:
        out = _speckle(out, rng, p["speckle"][1], p["speckle"][2])

    if rng.random() < p["jpeg"][0]:
        out = _jpeg(out, rng, p["jpeg"][1], p["jpeg"][2])

    return out


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "api_check")
    os.makedirs(out_dir, exist_ok=True)
    page = np.full((1400, 1000, 3), 245, dtype=np.uint8)
    cv2.putText(page, "INVOICE 1200.00", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    for i in range(20):
        cv2.line(page, (60, 240 + i * 40), (940, 240 + i * 40), (90, 90, 90), 2)
    for level in ("mild", "medium", "heavy"):
        d1 = degrade_page(page, seed=7, level=level)
        d2 = degrade_page(page, seed=7, level=level)
        assert (d1 == d2).all(), "degradation must be deterministic per seed"
        cv2.imwrite(os.path.join(out_dir, f"degrade_{level}.png"), d1)
        print(f"{level}: mean={d1.mean():.1f} std={d1.std():.1f} (deterministic OK)")
    print("degradation_document OK")
