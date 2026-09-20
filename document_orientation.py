"""
document_orientation.py — page orientation handling (EXIF, OSD, geometry).

Measured 2026-09-20 (this repo):
  * RapidOCR reads 180°/90° pages as garbage (CER 0.80-0.85 vs 0.09-0.39
    upright) while token confidences stay ~99 — OCR confidence cannot detect
    rotation, and RapidOCR's line classifier does not fix page-level 180°.
  * Tesseract OSD is exact on clean pages on the order of >=1500 px
    (0/90/180/270, confidence >= 5.7) but WRONG on small receipts even when
    upscaled (3/6 upright receipts reported 180° at confidence up to 4.8).

Policy (each rule exists because of the measurements above):
  * EXIF orientation is applied at load (`load_image_bgr`) — free, and the
    common case for phone photos (orientation 3 = 180° included).
  * OSD auto-rotation is applied ONLY for 90°/270° on pages with long side
    >= OSD_MIN_SIDE (receipts and small scans are excluded by construction).
    For these the detector geometry itself guarantees the axis; OSD only
    supplies the direction.
  * 180° is NEVER auto-rotated from OSD: the false-positive that rotated an
    upright large receipt 180° (conf 4.9) and the probe that found no GT-free
    text feature separating upright from 180° OCR (aggregate token stats were
    identical) make blind flipping destructive. OSD's 180° opinion is recorded
    as source `osd-180` and surfaces as `orientation_suspect`.
  * 90°/270° on small inputs also get the geometry `orientation_suspect`
    marker (no auto action: direction is ambiguous without OSD).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

OSD_MIN_SIDE = 1500      # long side, px — receipts are below this on purpose
OSD_MIN_CONF = 3.0       # clean pages score >=4.5; below this OSD is noise
VERTICAL_RATIO = 1.5     # token h/w above this counts as a vertical line
VERTICAL_FRACTION = 0.6  # ... and this fraction of tokens makes the page suspect


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
    """Decide the page rotation to apply before restore/OCR (see module docstring)."""
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
        # never flip on OSD alone (see module docstring); surface for review
        return RotationInfo(0, osd["confidence"], "osd-180", raw_angle=180)
    return RotationInfo(angle, osd["confidence"], "osd", raw_angle=angle)


def orientation_suspect(tokens: Sequence, min_tokens: int = 5) -> bool:
    """True when token geometry says the page is 90/270 (vertical text lines)."""
    if len(tokens) < min_tokens:
        return False
    vertical = 0
    for tok in tokens:
        x0, y0, x1, y1 = tok.bbox
        tw, th = max(1, x1 - x0), max(1, y1 - y0)
        if th / tw >= VERTICAL_RATIO:
            vertical += 1
    return vertical / len(tokens) >= VERTICAL_FRACTION


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
    print("document_orientation OK")
