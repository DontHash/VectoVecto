"""
document_router.py — mixed-page router: restore text, keep non-text (P5).

The restore step is tuned for text; applying it to logos, photos and
signatures smooths or invents detail. The router composites: restored pixels
inside a mask built from OCR text boxes (measured area coverage 0.975 on the
letterpress set), original pixels everywhere else. Photo regions can
optionally be enhanced by a caller-supplied upscaler; the default never
touches them.

No recognition changes: OCR still runs on the raw page (the measured best
stream); this only changes the *exported page*.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from document_ocr import Token


def text_mask_from_tokens(shape, tokens: Sequence[Token],
                          pad: float = 0.06) -> np.ndarray:
    """uint8 0/255 mask of token boxes, padded by `pad` of each box size."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for t in tokens:
        x0, y0, x1, y1 = t.bbox
        dx = int((x1 - x0) * pad)
        dy = int((y1 - y0) * pad)
        cv2.rectangle(mask, (max(0, x0 - dx), max(0, y0 - dy)),
                      (min(w - 1, x1 + dx), min(h - 1, y1 + dy)), 255, -1)
    return mask


def composite_regions(original: np.ndarray, restored: np.ndarray,
                      mask: np.ndarray, feather: int = 5) -> np.ndarray:
    """Restored inside `mask`, original outside; feathered seam."""
    m = mask.astype(np.float32) / 255.0
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather)
    m3 = m[..., None]
    out = (restored.astype(np.float32) * m3
           + original.astype(np.float32) * (1.0 - m3))
    return np.clip(out, 0, 255).astype(np.uint8)


def route_page(original: np.ndarray, restored: np.ndarray,
               tokens: Sequence[Token],
               regions: Optional[List[Dict]] = None,
               photo_upscaler: Optional[Callable[[np.ndarray], np.ndarray]] = None
               ) -> np.ndarray:
    """Composite the routed page. `regions` (synthetic eval sidecar) marks
    photo areas for the optional upscaler; without it, non-text is untouched."""
    mask = text_mask_from_tokens(original.shape, tokens)
    out = composite_regions(original, restored, mask)
    if photo_upscaler is not None and regions:
        h, w = out.shape[:2]
        for r in regions:
            if r.get("kind") != "photo":
                continue
            x0, y0, x1, y1 = r["bbox"]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = out[y0:y1, x0:x1]
            enhanced = photo_upscaler(patch)
            if enhanced is None:
                continue
            if enhanced.shape[:2] != patch.shape[:2]:
                enhanced = cv2.resize(enhanced, (patch.shape[1], patch.shape[0]))
            out[y0:y1, x0:x1] = enhanced
    return out
