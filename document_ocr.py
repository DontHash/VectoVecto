"""
document_ocr.py — OCR adapter registry for the document pipeline.

Backends (both offline, both optional at runtime):
  * rapidocr  — RapidOCR 3.x (PP-OCRv6 ONNX, Apache-2.0, models bundled) [default]
  * tesseract — Tesseract 5 via subprocess TSV output (Apache-2.0) [optional]

Design:
  * `Token` is the single unit: text, confidence 0-100, bbox, granularity
    (RapidOCR yields lines; Tesseract yields words), flags.
  * Never guess: flags are set from measured evidence (low conf, digit runs,
    digit re-pass disagreement).
  * `available_backends()` lets the harness/CLI degrade gracefully when a
    backend is not installed.

API:
    res = ocr_page(img_bgr, backend="rapidocr", lang=None) -> OCRResult
    backends = available_backends()  # ["rapidocr"] / ["rapidocr", "tesseract"]
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DIGIT_RUN_RE = re.compile(r"[0-9][0-9.,:\-/]*[0-9]|[0-9]")

# Measured on the frozen synthetic set (6 pages @300dpi, mild/medium/heavy):
#   rapidocr  raw            CER 0.0937
#   rapidocr  restored clahe CER 0.1424  (hurts -> feed raw)
#   tesseract raw            CER 0.3844
#   tesseract restored gray  CER 0.1749  (helps -> feed restored grayscale)
# Re-measure with `eval_document.py` whenever this table changes.
RECOMMENDED_STREAM: Dict[str, str] = {
    "rapidocr": "raw",
    "tesseract": "restore_gray",
}


@dataclass
class Token:
    text: str
    conf: float  # 0-100 (Tesseract convention; RapidOCR scores are *100)
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    granularity: str = "line"  # "line" | "word"
    backend: str = "?"
    flags: List[str] = field(default_factory=list)
    alt_text: Optional[str] = None  # second-stream reading, when it disagreed

    @property
    def has_digits(self) -> bool:
        return bool(DIGIT_RUN_RE.search(self.text))


@dataclass
class OCRResult:
    text: str
    tokens: List[Token]
    backend: str
    meta: Dict = field(default_factory=dict)


class OCRBackend:
    name = "base"

    def available(self) -> Tuple[bool, str]:
        return True, ""

    def run(self, img_bgr: np.ndarray, lang: Optional[str] = None) -> OCRResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RapidOCR (default)
# ---------------------------------------------------------------------------

class RapidOCRBackend(OCRBackend):
    name = "rapidocr"

    def __init__(self):
        self._engine = None
        self._error: Optional[str] = None

    def available(self) -> Tuple[bool, str]:
        try:
            import rapidocr  # noqa: F401
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"rapidocr not importable: {e}"

    def _ensure(self):
        if self._engine is None and self._error is None:
            try:
                from rapidocr import RapidOCR
                self._engine = RapidOCR()
            except Exception as e:  # noqa: BLE001
                self._error = str(e)
        if self._engine is None:
            raise RuntimeError(f"RapidOCR unavailable: {self._error}")

    def run(self, img_bgr: np.ndarray, lang: Optional[str] = None) -> OCRResult:
        self._ensure()
        t0 = time.time()
        res = self._engine(img_bgr)
        tokens: List[Token] = []
        boxes = getattr(res, "boxes", None)
        txts = getattr(res, "txts", None) or ()
        scores = getattr(res, "scores", None) or ()
        if boxes is not None:
            for box, txt, score in zip(boxes, txts, scores):
                arr = np.asarray(box)
                x0, y0 = int(arr[:, 0].min()), int(arr[:, 1].min())
                x1, y1 = int(arr[:, 0].max()), int(arr[:, 1].max())
                tokens.append(Token(text=str(txt), conf=float(score) * 100.0,
                                    bbox=(x0, y0, x1, y1), granularity="line",
                                    backend=self.name))
        text = "\n".join(t.text for t in tokens)
        return OCRResult(text=text, tokens=tokens, backend=self.name,
                         meta={"seconds": round(time.time() - t0, 3),
                               "n_tokens": len(tokens), "lang": lang or "default"})


# ---------------------------------------------------------------------------
# Tesseract (optional)
# ---------------------------------------------------------------------------

class TesseractBackend(OCRBackend):
    name = "tesseract"

    @staticmethod
    def _find() -> Optional[str]:
        exe = shutil.which("tesseract")
        if exe:
            return exe
        for cand in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                     "/usr/bin/tesseract", "/usr/local/bin/tesseract",
                     "/opt/homebrew/bin/tesseract"):
            if os.path.exists(cand):
                return cand
        return None

    def available(self) -> Tuple[bool, str]:
        exe = self._find()
        if not exe:
            return False, ("tesseract binary not found "
                           "(install: winget install -e --id tesseract-ocr.tesseract)")
        return True, exe

    def _parse_tsv(self, tsv: str) -> List[Token]:
        tokens: List[Token] = []
        header: Optional[List[str]] = None
        for line in tsv.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < 12:
                continue
            try:
                level = int(parts[0])
                conf = float(parts[10])
                text = parts[11].strip()
            except ValueError:
                continue
            if level != 5 or not text:  # level 5 == word
                continue
            left, top, w, h = (int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]))
            tokens.append(Token(text=text, conf=conf, bbox=(left, top, left + w, top + h),
                                granularity="word", backend=self.name))
        return tokens

    def run(self, img_bgr: np.ndarray, lang: Optional[str] = None) -> OCRResult:
        exe = self._find()
        if not exe:
            raise RuntimeError("tesseract binary not found")
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.png")
            cv2.imwrite(inp, img_bgr)
            cmd = [exe, inp, "stdout", "-l", lang or "eng", "--psm", "6", "tsv"]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(f"tesseract failed: {proc.stderr[-300:]}")
            tokens = self._parse_tsv(proc.stdout)
        text = "\n".join(t.text for t in tokens)
        return OCRResult(text=text, tokens=tokens, backend=self.name,
                         meta={"seconds": round(time.time() - t0, 3),
                               "n_tokens": len(tokens), "lang": lang or "eng"})


# ---------------------------------------------------------------------------
# registry + flagging helpers
# ---------------------------------------------------------------------------

_BACKENDS = {"rapidocr": RapidOCRBackend, "tesseract": TesseractBackend}


def available_backends() -> List[str]:
    out = []
    for name, cls in _BACKENDS.items():
        ok, _why = cls().available()
        if ok:
            out.append(name)
    return out


def get_backend(name: str) -> OCRBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown OCR backend {name!r}; have {sorted(_BACKENDS)}")
    backend = _BACKENDS[name]()
    ok, why = backend.available()
    if not ok:
        raise RuntimeError(f"OCR backend {name!r} unavailable: {why}")
    return backend


def ocr_page(img_bgr: np.ndarray, backend: str = "rapidocr",
             lang: Optional[str] = None, conf_threshold: float = 60.0,
             recheck_digits: bool = False) -> OCRResult:
    """Run OCR and attach flags. `recheck_digits` needs a second backend-free pass
    (implemented in Phase C as recognition-only re-run; here it is a hook)."""
    result = get_backend(backend).run(img_bgr, lang=lang)
    for tok in result.tokens:
        if tok.conf < conf_threshold:
            tok.flags.append("low_conf")
        if tok.has_digits and tok.conf < 80.0:
            tok.flags.append("digit_uncertain")
    result.meta["conf_threshold"] = conf_threshold
    result.meta["flagged"] = sum(1 for t in result.tokens if t.flags)
    return result


# ---------------------------------------------------------------------------
# do-not-hallucinate gate: two streams must agree on digits
# ---------------------------------------------------------------------------

def _digits_of(text: str) -> str:
    return "".join(re.findall(r"\d+", text))


def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def compare_digit_streams(primary: OCRResult, alt: OCRResult,
                          center_tolerance: float = 1.0) -> int:
    """Flag primary digit tokens where the nearest alt token reads different digits.

    Tolerance is in units of the primary token's height. Never picks a winner —
    it only records `digit_conflict` and stores `alt_text` for the UI.
    """
    conflicts = 0
    for tok in primary.tokens:
        if not tok.has_digits:
            continue
        cx, cy = _center(tok.bbox)
        tol = max(8.0, (tok.bbox[3] - tok.bbox[1]) * center_tolerance)
        best, best_d = None, float("inf")
        for other in alt.tokens:
            if not other.has_digits:
                continue
            ox, oy = _center(other.bbox)
            d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, other
        if best is not None and best_d <= tol and _digits_of(tok.text) != _digits_of(best.text):
            if "digit_conflict" not in tok.flags:
                tok.flags.append("digit_conflict")
            tok.alt_text = best.text
            conflicts += 1
    return conflicts


def ocr_page_dual(img_raw: np.ndarray, img_alt: Optional[np.ndarray] = None,
                  backend: str = "rapidocr", lang: Optional[str] = None,
                  conf_threshold: float = 60.0) -> OCRResult:
    """OCR the recommended stream as primary; the other stream audits digits.

    stream choice follows RECOMMENDED_STREAM (measured, see module docstring):
      rapidocr  -> primary = raw,      auditor = restored
      tesseract -> primary = restored, auditor = raw
    """
    alt_img = img_alt if img_alt is not None else img_raw
    if backend == "tesseract":
        primary_img, audit_img = alt_img, img_raw
    else:
        primary_img, audit_img = img_raw, alt_img

    primary = ocr_page(primary_img, backend=backend, lang=lang,
                       conf_threshold=conf_threshold)
    conflicts = 0
    if audit_img is not None and audit_img is not img_raw or backend == "tesseract":
        try:
            audit = ocr_page(audit_img, backend=backend, lang=lang,
                             conf_threshold=conf_threshold)
            conflicts = compare_digit_streams(primary, audit)
        except Exception as e:  # noqa: BLE001
            primary.meta["audit_error"] = str(e)
    primary.meta["digit_conflicts"] = conflicts
    primary.meta["primary_stream"] = "restored" if backend == "tesseract" else "raw"
    return primary


if __name__ == "__main__":
    import sys

    print("available backends:", available_backends())
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path and os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        for backend in available_backends():
            res = ocr_page(img, backend=backend)
            print(f"\n[{backend}] {len(res.tokens)} tokens, {res.meta}")
            for tok in res.tokens[:4]:
                print(f"  {tok.conf:6.1f} {tok.text!r} flags={tok.flags}")
        print("\nfull text:\n", res.text)
    else:
        page = np.full((300, 900, 3), 255, dtype=np.uint8)
        cv2.putText(page, "INVOICE 1200.00 TOTAL", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
        for backend in available_backends():
            res = ocr_page(page, backend=backend)
            print(f"[{backend}] {res.text!r}")
