"""
document_restore.py — classical document restoration core (Phase B).

Problem it solves: phone photos and scans of pages are skewed, unevenly lit,
soft, and noisy — which hurts OCR more than any missing megapixels. The fixes
are classical and deterministic (ScanTailor-style), not GANs:

    orient -> downscale -> deskew -> illumination flatten -> light denoise
           -> OCR stream (Sauvola) + display stream (clean grayscale)

API (numpy in -> numpy out, harness contract):
    restore_document(img_bgr, *, scale=1) -> dict with keys
        display_bgr : cleaned grayscale rendered as BGR (what the user sees)
        ocr_bgr     : binarized/contrast-normalized sibling (what OCR eats)
        debug       : {skew_angle, scale_factor, illum, binary, seconds}

No sr_engine / no neural models in this path. No network.
"""
from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np

MAX_SIDE = 2500  # phone photos beyond this gain nothing for OCR, only latency
SKEW_LIMIT = 8.0
SKEW_STEP = 0.5
SKEW_PROBE_SIDE = 800


# ---------------------------------------------------------------------------
# stages (each independently testable)
# ---------------------------------------------------------------------------

def fit_max_side(img: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if not max_side or m <= max_side:
        return img, 1.0
    f = max_side / m
    out = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                     interpolation=cv2.INTER_AREA)
    return out, f


def estimate_skew(gray: np.ndarray, min_gain: float = 1.15) -> float:
    """Projection-profile deskew: the rotation that maximizes the variance of
    horizontal ink projection (text rows align -> sharper peaks).

    Guard: if the best angle does not beat the 0-degree score by `min_gain`,
    return 0.0. Shadows, page borders and perspective otherwise create bogus
    maxima (observed: -8 deg on a shadowed page -> destroyed OCR).
    """
    h, w = gray.shape[:2]
    probe, _ = fit_max_side(gray, SKEW_PROBE_SIDE)
    probe = cv2.GaussianBlur(probe, (0, 0), 1.0)
    ph, pw = probe.shape[:2]
    _, binary = cv2.threshold(probe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.float32)

    scores = {}
    for angle in np.arange(-SKEW_LIMIT, SKEW_LIMIT + 1e-6, SKEW_STEP):
        m = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        rot = cv2.warpAffine(binary, m, (pw, ph), flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1)
        scores[round(float(angle), 3)] = float(np.var(proj))

    best_angle = max(scores, key=scores.get)
    zero_score = scores.get(0.0, 0.0)
    if scores[best_angle] < zero_score * min_gain:
        return 0.0
    # sub-step refinement around the winner
    for cand in np.arange(best_angle - SKEW_STEP, best_angle + SKEW_STEP + 1e-6, SKEW_STEP / 2):
        if cand in scores or abs(cand) > SKEW_LIMIT:
            continue
        m = cv2.getRotationMatrix2D((pw / 2, ph / 2), cand, 1.0)
        rot = cv2.warpAffine(binary, m, (pw, ph), flags=cv2.INTER_NEAREST)
        scores[round(float(cand), 3)] = float(np.var(rot.sum(axis=1)))
    best_angle = max(scores, key=scores.get)
    return best_angle  # candidate rotation that aligns text rows == correction angle


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def flatten_illumination(gray: np.ndarray, ksize: int = 41,
                         work_scale: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Divide by a large-kernel background estimate to remove shadows/gradients.

    The background is estimated at 1/work_scale resolution (median blur cost at
    full A4@300dpi was ~10 s; this is ~0.2 s and visually identical for smooth
    lighting fields).
    """
    h, w = gray.shape[:2]
    k = max(3, ksize | 1)
    if work_scale > 1 and min(h, w) > 4 * work_scale:
        small = cv2.resize(gray, (max(1, w // work_scale), max(1, h // work_scale)),
                           interpolation=cv2.INTER_AREA)
        bk = max(3, (k // work_scale) | 1)
        bg_small = cv2.medianBlur(small, bk)
        background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        background = cv2.medianBlur(gray, k)
    background = cv2.GaussianBlur(background, (0, 0), k / 3.0)
    target = float(np.percentile(background, 90))
    norm = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0) * target
    return np.clip(norm, 0, 255).astype(np.uint8), background


def light_denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving clean-up that does not smear strokes."""
    try:
        return cv2.ximgproc.guidedFilter(gray, gray, radius=4, eps=60.0)
    except Exception:
        return cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=5)


def sauvola_binary(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    from skimage.filters import threshold_sauvola
    t = threshold_sauvola(gray, window_size=window, k=k)
    return ((gray > t).astype(np.uint8) * 255)


def clahe_gray(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)


def build_ocr_stream(clean_gray: np.ndarray, mode: str = "clahe") -> np.ndarray:
    """What OCR actually eats. Measured per mode on the harness; do not guess."""
    if mode == "sauvola":
        stream = sauvola_binary(clean_gray)
    elif mode == "gray":
        stream = clean_gray
    elif mode == "clahe":
        stream = clahe_gray(clean_gray)
    else:
        raise ValueError(f"unknown ocr_stream {mode!r}")
    return cv2.cvtColor(stream, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def restore_document(img_bgr: np.ndarray, *, scale: int = 1,
                     max_side: int = MAX_SIDE, ocr_stream: str = "clahe",
                     deskew: bool = True) -> Dict:
    """Full classical restore. `deskew=False` keeps geometry identical to the
    input (callers that overlay raw-OCR boxes on the display need this)."""
    t0 = time.time()
    if img_bgr.ndim == 2:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        img = img_bgr if img_bgr.shape[2] == 3 else img_bgr[:, :, :3]

    img, f = fit_max_side(img, max_side)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flat, background = flatten_illumination(gray)
    flat = light_denoise(flat)

    skew = estimate_skew(flat) if deskew else 0.0
    clean = rotate(flat, skew)
    binary = sauvola_binary(clean)

    if scale == 2:
        h, w = clean.shape[:2]
        clean = cv2.resize(clean, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        binary = cv2.resize(binary, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    display_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    ocr_bgr = build_ocr_stream(clean, ocr_stream)
    return {
        "display_bgr": display_bgr,
        "ocr_bgr": ocr_bgr,
        "debug": {
            "skew_angle": round(float(skew), 3),
            "scale_factor": f,
            "illum": background,
            "binary": binary,
            "ocr_stream": ocr_stream,
            "deskew": deskew,
            "seconds": round(time.time() - t0, 3),
        },
    }


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import doc_data
    from degradation_document import degrade_page

    page, gt = doc_data.render_synthetic_invoice(seed=5)
    deg = degrade_page(page, seed=5, level="heavy")
    out = restore_document(deg)
    print(f"skew={out['debug']['skew_angle']} scale={out['debug']['scale_factor']} "
          f"seconds={out['debug']['seconds']}")
    os.makedirs("out/api_check", exist_ok=True)
    cv2.imwrite("out/api_check/restore_display.png", out["display_bgr"])
    cv2.imwrite("out/api_check/restore_ocr.png", out["ocr_bgr"])

    from document_ocr import ocr_page
    raw_cer = None
    try:
        import doc_metrics
        raw = ocr_page(deg, backend="rapidocr")
        res = ocr_page(out["ocr_bgr"], backend="rapidocr")
        raw_cer = doc_metrics.cer(gt, raw.text)
        print(f"CER raw={raw_cer:.4f} restored={doc_metrics.cer(gt, res.text):.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"ocr check skipped: {e}")
    print("document_restore OK")
