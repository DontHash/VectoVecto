"""
document_orientation.py — page orientation handling.

Corrected findings 2026-09-20 (the first P3 pass misread reverse line ORDER as
garbage TEXT — order-sensitive CER hid it):
  * RapidOCR reads 180° pages perfectly, line for line (cls flips each crop,
    conf 0.9999); the lines come out in reversed order because detection walks
    the flipped image top-to-bottom. Re-OCR after a 180° image rotation is
    clean (synthetic CER 0.000).
  * 90°/270° pages: detection perspective-unrotates the line boxes, so the
    TEXT is also largely correct, but the line classifier votes 0/180 per box
    ambiguously (fractions 0.12-0.55) and vertical boxes dominate (h/w >= 1.5
    on 0.91-1.00 of tokens vs <0.6 on upright receipts).
  * Page-level evidence that works (measured):
      - cls votes (share of lines classed 180; share with score >= 0.9):
        flipped 0.73-1.00 / 0.45-1.00, upright <=0.50 / <=0.25; action needs
        frac >= 0.6 AND hi >= 0.4 (0 false flips / 60 upright pages, 59/60
        flipped detected on SROIE+CORD). Classifier-hard photos vote high BOTH
        ways and land in `vote-ambiguous` (flag, never act);
      - vertical token fraction: >=0.91 sideways vs <0.6 upright (0/30 SROIE);
      - a 90°-clockwise probe pass resolves the axis (probe verticality drops
        to 0) and its votes then pick the direction (270° vs 90°).
  * OSD (tesseract) stays only as a fallback: no cls votes (tesseract backend)
    or inconclusive probes. RapidOCR never needs it.

Policy:
  * EXIF orientation is applied at load (`load_image_bgr`) — phone photos.
  * Evidence-first: OCR once, then votes/verticality decide 0/90/180/270
    (see `infer_angle`); a rotation re-runs the pipeline once on the rotated
    page. No blind flips: every decision rests on per-line classifier votes
    or measured geometry.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

OSD_MIN_SIDE = 1500      # long side, px — OSD fallback only trusts large pages
OSD_MIN_CONF = 3.0       # clean pages score >=4.5; below this OSD is noise
VERTICAL_RATIO = 1.5     # token h/w above this counts as a vertical line
VERTICAL_FRACTION = 0.6  # ... and this fraction makes the page sideways
VERTICAL_ACTION = 0.6    # measured: sideways 0.91-1.00, upright <0.6 (0/30)
VOTE180_FRAC = 0.6       # flipped frac180: 0.73-1.00 | upright: <=0.50 (60 pages)
VOTE180_HI = 0.4         # flipped hi-conf 180: 0.45-1.00 | upright: <=0.25
VOTE180_AMBIG_FRAC = 0.5  # >= this but not actionable -> suspect, never act
MIN_TOKENS = 5           # below this there is no reliable page-level evidence


@dataclass
class RotationInfo:
    angle: int = 0            # clockwise degrees applied
    confidence: float = 0.0
    source: str = "off"       # exif | osd | skipped-small | low-confidence | ...
    raw_angle: int = 0        # OSD's opinion even when not applied

    def as_dict(self) -> Dict:
        return asdict(self)


_ROTATE_RE = re.compile(r"^Rotate:\s*(\d+)", re.MULTILINE)
_ORIENT_RE = re.compile(r"Orientation in degrees:\s*(\d+)")
_CONF_RE = re.compile(r"Orientation confidence:\s*([\d.]+)")
_SCRIPT_RE = re.compile(r"Script:\s*(\S+)")


def parse_osd(stdout: str) -> Optional[Dict]:
    """Parse tesseract --psm 0 output. `rotate` = clockwise degrees to correct."""
    rotate_m = _ROTATE_RE.search(stdout or "")
    orient_m = _ORIENT_RE.search(stdout or "")
    if rotate_m is None and orient_m is None:
        return None
    orientation = int(orient_m.group(1)) if orient_m else 0
    rotate = int(rotate_m.group(1)) if rotate_m else (360 - orientation) % 360
    conf_m = _CONF_RE.search(stdout or "")
    script_m = _SCRIPT_RE.search(stdout or "")
    return {
        "orientation": orientation % 360,
        "rotate": rotate % 360,
        "confidence": float(conf_m.group(1)) if conf_m else 0.0,
        "script": script_m.group(1) if script_m else "",
    }


def _find_tesseract() -> Optional[str]:
    from document_ocr import TesseractBackend
    return TesseractBackend._find()


def osd_read(img_bgr: np.ndarray) -> Optional[Dict]:
    exe = _find_tesseract()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "page.png")
        cv2.imwrite(path, img_bgr)
        try:
            proc = subprocess.run([exe, path, "stdout", "--psm", "0"],
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=180)
        except Exception:  # noqa: BLE001
            return None
    return parse_osd(proc.stdout or "")


def rotate_bgr(img: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 (other values are rounded to 90)."""
    deg = int(round(degrees / 90.0) * 90) % 360
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def detect_rotation(img_bgr: np.ndarray) -> RotationInfo:
    """OSD-only fallback (tesseract backend / inconclusive OCR evidence).

    Kept for backends without a line classifier; the rapidocr path uses
    `vertical_fraction` + `line_orientation_votes` instead (see module docstring).
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) < OSD_MIN_SIDE:
        return RotationInfo(0, 0.0, "skipped-small")
    if _find_tesseract() is None:
        return RotationInfo(0, 0.0, "no-tesseract")
    osd = osd_read(img_bgr)
    if osd is None:
        return RotationInfo(0, 0.0, "osd-failed")
    if osd["confidence"] < OSD_MIN_CONF:
        return RotationInfo(0, osd["confidence"], "low-confidence",
                            raw_angle=osd["rotate"] % 360)
    angle = osd["rotate"] % 360
    if angle == 180:
        # OSD false-positived 180 on a large upright receipt (conf 4.9); the
        # rapidocr evidence path handles 180 properly, this fallback does not.
        return RotationInfo(0, osd["confidence"], "osd-180", raw_angle=180)
    return RotationInfo(angle, osd["confidence"], "osd", raw_angle=angle)


def vertical_fraction(tokens: Sequence, min_tokens: int = MIN_TOKENS) -> float:
    """Share of tokens whose bbox is taller than wide (sideways text lines)."""
    if len(tokens) < min_tokens:
        return 0.0
    vertical = 0
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        tw, th = max(1, x1 - x0), max(1, y1 - y0)
        if th / tw >= VERTICAL_RATIO:
            vertical += 1
    return vertical / len(tokens)


def orientation_suspect(tokens: Sequence, min_tokens: int = MIN_TOKENS) -> bool:
    """True when token geometry still says the page is 90/270."""
    return vertical_fraction(tokens, min_tokens) >= VERTICAL_ACTION


def infer_angle(tokens: Sequence, votes: Optional[Tuple[float, float, int]], *,
                auto_rotate: bool = True) -> RotationInfo:
    """Evidence-only orientation decision (no OCR, no I/O).

    `votes` = (frac180, hi_conf_180, n_lines) from the backend, or None when it
    has no line classifier. Measured separation (60 real pages, both ways):
      * flipped: frac180 0.73-1.00, hi-conf 0.45-1.00; vertical ~0
      * upright: frac180 <=0.50, hi-conf <=0.25 (except classifier-hard pages
        that vote high BOTH ways - excluded by requiring hi-conf too);
      * sideways: >=0.91 of tokens have tall boxes; votes are mixed.
    The action rule (frac >= 0.6 AND hi >= 0.4) gave 0 false flips / 60 upright
    and 59/60 flipped detected; the miss and the both-ways-confused pages get
    the `vote-ambiguous` suspect flag instead of an action.

    Returns source:
      votes-180      -> rotate 180
      vote-ambiguous -> classifier cannot be trusted; flag, do not act
      probe-needed   -> rotate 90cw and re-read to resolve axis/direction
      upright        -> no action
      off            -> auto-rotate disabled; raw_angle carries the suggestion
    """
    vfrac = vertical_fraction(tokens)
    frac = votes[0] if votes else 0.0
    hi = votes[1] if votes else 0.0
    authoritative = len(tokens) >= MIN_TOKENS and votes is not None
    if authoritative and frac >= VOTE180_FRAC and hi >= VOTE180_HI:
        if auto_rotate:
            return RotationInfo(180, round(frac, 3), "votes-180", raw_angle=180)
        return RotationInfo(0, round(frac, 3), "off", raw_angle=180)
    if vfrac >= VERTICAL_ACTION:
        if auto_rotate:
            return RotationInfo(0, round(vfrac, 3), "probe-needed", raw_angle=90)
        return RotationInfo(0, round(vfrac, 3), "off", raw_angle=90)
    if authoritative and frac >= VOTE180_AMBIG_FRAC:
        if auto_rotate:
            return RotationInfo(0, round(frac, 3), "vote-ambiguous", raw_angle=180)
        return RotationInfo(0, round(frac, 3), "off", raw_angle=180)
    return RotationInfo(0, round(frac, 3), "upright")


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """cv2.imread with EXIF orientation applied (phone photos)."""
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            arr = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        return cv2.imread(path, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    sample = ("Page number: 0\n"
              "Orientation in degrees: 90\n"
              "Rotate: 270\n"
              "Orientation confidence: 6.44\n"
              "Script: Latin\n"
              "Script confidence: 2.67\n")
    parsed = parse_osd(sample)
    assert parsed["rotate"] == 270 and parsed["orientation"] == 90
    assert parsed["confidence"] == 6.44 and parsed["script"] == "Latin"
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    assert rotate_bgr(img, 90).shape[:2] == (300, 100)
    assert rotate_bgr(img, 180).shape[:2] == (100, 300)
    assert rotate_bgr(img, 270).shape[:2] == (300, 100)
    assert detect_rotation(img).source == "skipped-small"

    class _Tok:
        def __init__(self, box):
            self.bbox = box
    flat = [_Tok((0, 0, 100, 20))] * 6
    tall = [_Tok((0, 0, 20, 100))] * 6
    assert infer_angle(flat, (0.1, 0.1, 6)).source == "upright"
    assert infer_angle(flat, (1.0, 1.0, 6)).angle == 180
    assert infer_angle(flat, (1.0, 1.0, 6), auto_rotate=False).raw_angle == 180
    assert infer_angle(flat, (0.8, 0.2, 6)).source == "vote-ambiguous"
    assert infer_angle(tall, (0.4, 0.1, 6)).source == "probe-needed"
    assert infer_angle(tall, (0.4, 0.1, 6), auto_rotate=False).source == "off"
    assert infer_angle(flat[:3], (1.0, 1.0, 3)).source == "upright"
    print("document_orientation OK")
