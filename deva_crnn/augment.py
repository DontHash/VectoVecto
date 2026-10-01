"""
augment.py — print-degradation augmentation for line crops (local use).

Kept out of `data.py` so the training path never imports OpenCV: the Vertex
PyTorch container has torch/PIL/numpy but not cv2.

`level="light"` is the attempt-1 augmentation (blur/noise/JPEG/brightness).
`level="heavy"` adds the artifacts that actually separate a rendered line from
a scanned one: ink spread/erosion, resolution loss (downscale-upscale), uneven
illumination, stronger blur/noise/JPEG and a slight rotation.
`level="xheavy"` is the stress set for few-shot real crops (N5 digit lines):
affine + perspective + elastic geometry, local shadow fields, gamma and faded
print, motion blur, salt-and-pepper and speckle noise, tighter crop jitter.
Existing levels are frozen; new work extends by adding a level.
"""
from __future__ import annotations

import cv2
import numpy as np


def augment_line(img_bgr: np.ndarray, rng: np.random.Generator,
                 level: str = "light") -> np.ndarray:
    """Deterministic print-degradation augmentation (given `rng`)."""
    if level == "xheavy":
        return _xheavy(img_bgr, rng)
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


def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random-angle linear motion blur (hand-held phone captures)."""
    k = int(rng.integers(3, 8)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    m = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    total = float(kernel.sum())
    if total <= 0:
        return img
    return cv2.filter2D(img, -1, kernel / total)


def _elastic(img: np.ndarray, rng: np.random.Generator, alpha: float,
             sigma: float) -> np.ndarray:
    """Mild smooth displacement field (paper curl, lens distortion)."""
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0),
        sigmaX=sigma, sigmaY=sigma) * alpha
    dy = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0),
        sigmaX=sigma, sigmaY=sigma) * alpha
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + dx
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) + dy
    return cv2.remap(img, xs, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _shadow_field(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smooth local darkening (a hand/phone shadow across the page)."""
    h, w = img.shape[:2]
    low = rng.uniform(0.55, 1.0, (4, 6)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(2.0, w / 40.0))
    return img * field[:, :, None]


def _xheavy(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stress augmentation for few-shot real crops (N5 digit lines).

    Chain: geometry (affine/perspective/elastic) -> ink -> resolution ->
    illumination (shadow fields, gamma, fade) -> optics (motion/gaussian
    blur) -> sensor (gaussian/salt-pepper/speckle) -> JPEG last. Every step
    is applied with its own probability and bounded so the label is
    preserved: no flips, rotation <= 2.5 deg, displacement <= ~3 px.
    """
    out = img_bgr.astype(np.float32)
    h, w = out.shape[:2]

    # --- geometry ---
    if rng.random() < 0.7:
        ang = float(rng.uniform(-2.5, 2.5))
        scale = float(rng.uniform(0.92, 1.08))
        shear = float(rng.uniform(-0.035, 0.035))
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, scale)
        m[0, 1] += shear
        m[0, 2] += float(rng.uniform(-0.02, 0.02)) * w
        m[1, 2] += float(rng.uniform(-0.02, 0.02)) * h
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        j = 0.015
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + (rng.uniform(-j, j, src.shape).astype(np.float32)
                     * np.float32([w, h]))
        m = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, m, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3 and h >= 24 and w >= 40:
        out = _elastic(out, rng, alpha=float(rng.uniform(1.5, 3.0)),
                       sigma=float(rng.uniform(6.0, 9.0)))

    # --- ink ---
    if rng.random() < 0.6:
        k = (np.ones((3, 3), np.uint8) if rng.random() < 0.4
             else np.ones((2, 2), np.uint8))
        if rng.random() < 0.5:
            out = cv2.dilate(out, k, iterations=1).astype(np.float32)
        else:
            out = cv2.erode(out, k, iterations=1).astype(np.float32)

    # --- resolution ---
    if rng.random() < 0.55:
        f = float(rng.uniform(0.4, 0.75))
        small = cv2.resize(out, (max(4, int(w * f)), max(4, int(h * f))),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.45:
        fx = float(rng.uniform(0.6, 1.5))
        out = cv2.resize(out, (max(4, int(w * fx)), h),
                         interpolation=cv2.INTER_LINEAR)
        h, w = out.shape[:2]

    # --- illumination ---
    if rng.random() < 0.5:
        out = _shadow_field(out, rng)
    if rng.random() < 0.4:
        ramp = np.linspace(float(rng.uniform(0.7, 0.9)),
                           float(rng.uniform(1.0, 1.12)), w,
                           dtype=np.float32)
        out = out * ramp[None, :, None]
    if rng.random() < 0.4:
        gamma = float(rng.uniform(0.6, 1.6))
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    if rng.random() < 0.25:
        out = out * 0.55 + 245.0 * 0.45  # faded print toward paper tone
    out = np.clip(out * float(rng.uniform(0.7, 1.2)) +
                  float(rng.uniform(-20, 20)), 0, 255)

    # --- optics / sensor ---
    if rng.random() < 0.3:
        out = _motion_blur(out, rng)
    if rng.random() < 0.8:
        out = cv2.GaussianBlur(out, (0, 0),
                               sigmaX=float(rng.uniform(0.4, 2.6)))
    if rng.random() < 0.75:
        out = out + rng.normal(0.0, float(rng.uniform(3, 22)), out.shape)
    if rng.random() < 0.25:
        p = float(rng.uniform(0.002, 0.012))
        mask = rng.random((out.shape[0], out.shape[1]))
        dark = (mask < p * 0.5)[:, :, None]
        bright = ((mask >= p * 0.5) & (mask < p))[:, :, None]
        out = np.where(dark, 0.0, np.where(bright, 255.0, out))
    if rng.random() < 0.2:
        out = out * (1.0 + rng.normal(0.0, 0.06, out.shape))
    out = np.clip(out, 0, 255).astype(np.uint8)

    # --- compression last ---
    if rng.random() < 0.75:
        q = int(rng.integers(20, 66))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out
