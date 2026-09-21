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

DIGIT_RUN_RE = re.compile(r"[0-9\u0966-\u096f][0-9\u0966-\u096f.,:\-/]*"
                          r"[0-9\u0966-\u096f]|[0-9\u0966-\u096f]")

# User-facing language codes -> RapidOCR rec lang_type ("default" = shipped
# PP-OCRv6 model). Devanagari needs the PP-OCRv5 rec model (PP-OCRv6 does not
# ship that script); it covers both Nepali and Hindi.
LANG_ALIASES: Dict[str, str] = {
    "": "default", "en": "default", "eng": "default", "english": "default",
    "latin": "default",
    "ne": "devanagari", "nep": "devanagari", "nepali": "devanagari",
    "hi": "devanagari", "hin": "devanagari", "hindi": "devanagari",
    "devanagari": "devanagari",
}


def normalize_lang(lang: Optional[str]) -> str:
    """Map a user language code to a RapidOCR rec lang_type."""
    if not lang:
        return "default"
    return LANG_ALIASES.get(lang.strip().lower(), "default")

# Measured on the frozen synthetic set (6 pages @300dpi, mild/medium/heavy):
#   rapidocr  raw            CER 0.0937
#   rapidocr  restored clahe CER 0.1424  (hurts)
#   tesseract raw            CER 0.3844
#   tesseract restored gray  CER 0.1749  (helps -> feed restored grayscale)
# Open item (P4b): newer reports show rapidocr restored_gray at CER 0.0150 on
# the synthetic set and marginally better on SROIE (0.3618 vs 0.3635) and CORD
# (0.5748 vs 0.5872), but bagCER disagrees on synthetics (0.1049 vs 0.0269) -
# flagged for a proper re-measure before any change to this table.
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
    repass_text: Optional[str] = None  # recognition-only re-read on a 2x crop
    repass_conf: Optional[float] = None
    cal_conf: Optional[float] = None  # calibrated confidence (0-100), when fitted

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

    def recognize_crop(self, crop_bgr: np.ndarray, lang: Optional[str] = None) -> Tuple[str, float]:
        """Recognition-only re-read of a single crop (no detection). Used by the
        digit re-pass. Returns (text, conf 0-100)."""
        raise NotImplementedError

    def line_orientation_votes(self, img_bgr: np.ndarray,
                               result: OCRResult,
                               lang: Optional[str] = None) -> Optional[Tuple[float, float, int]]:
        """Page-flip evidence from the PP-LCNet line classifier: (frac180, hi180, n).

        `frac180` = share of line crops classed 180; `hi180` = share classed 180
        with score >= 0.9. Measured on 60 real pages, both orientations
        (2026-09-20): flipped pages score frac 0.73-1.00 / hi 0.45-1.00 while
        upright pages stay below frac 0.5 / hi 0.25 - except classifier-hard
        pages (blurry CORD photos) that vote high BOTH ways; the action rule in
        `document_orientation.infer_angle` (frac >= 0.6 AND hi >= 0.4) keeps
        those out (0 false flips / 60 upright, 59/60 flipped detected).
        """
        return None


# ---------------------------------------------------------------------------
# RapidOCR (default)
# ---------------------------------------------------------------------------

def _dml_enabled() -> bool:
    """RAPIDOCR_USE_DML: 'auto' (default) uses DirectML when available.

    Measured on the RTX 2050: CPU ~9-15 s/page, DirectML 25 s first call
    (kernel compile) then **0.78 s/page**. Worth a one-time warm-up.
    """
    env = os.environ.get("RAPIDOCR_USE_DML", "auto").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


class RapidOCRBackend(OCRBackend):
    name = "rapidocr"

    def __init__(self):
        self._engine = None          # default-language engine (also cls votes)
        self._lang_engines: Dict[str, object] = {}
        self._error: Optional[str] = None
        self.using_dml = False

    def available(self) -> Tuple[bool, str]:
        try:
            import rapidocr  # noqa: F401
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"rapidocr not importable: {e}"

    def _base_params(self) -> Dict:
        params: Dict = {"Global.log_level": os.environ.get("RAPIDOCR_LOG_LEVEL", "error")}
        self.using_dml = _dml_enabled()
        if self.using_dml:
            params["EngineConfig.onnxruntime.use_dml"] = True
        max_side = os.environ.get("RAPIDOCR_MAX_SIDE")
        if max_side:
            params["Global.max_side_len"] = int(max_side)
        return params

    def _ensure(self):
        if self._engine is None and self._error is None:
            try:
                from rapidocr import RapidOCR
                self._engine = RapidOCR(params=self._base_params())
                if self.using_dml:
                    self._warmup(self._engine)
            except Exception as e:  # noqa: BLE001
                self._error = str(e)
        if self._engine is None:
            raise RuntimeError(f"RapidOCR unavailable: {self._error}")

    def _engine_for(self, lang: Optional[str]):
        """Engine for a language code; devanagari uses the PP-OCRv5 rec model
        (PP-OCRv6 does not ship that script) and is cached per language."""
        key = normalize_lang(lang)
        if key == "default":
            self._ensure()
            return self._engine
        self._ensure()
        eng = self._lang_engines.get(key)
        if eng is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import ModelType, OCRVersion
            params = self._base_params()
            if key == "devanagari":
                params["Rec.lang_type"] = "devanagari"
                params["Rec.ocr_version"] = OCRVersion("PP-OCRv5")
                params["Rec.model_type"] = ModelType("mobile")
                # The 0/180 textline classifier is trained on Chinese/English
                # and flips Devanagari crops: measured page CER 0.537 with it,
                # 0.039 without (isolated crops read perfectly either way).
                params["Global.use_cls"] = False
            self._lang_engines[key] = eng = RapidOCR(params=params)
            if self.using_dml:
                self._warmup(eng)
        return eng

    def _warmup(self, engine):
        """Pay DirectML's kernel-compile cost once (first call is ~25 s, then <1 s)."""
        t0 = time.time()
        warm = np.full((64, 320, 3), 255, dtype=np.uint8)
        cv2.putText(warm, "warmup 123", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        try:
            engine(warm)
            print(f"[document_ocr] RapidOCR DirectML warm-up: {time.time() - t0:.1f}s "
                  f"(subsequent pages ~0.8s)")
        except Exception as e:  # noqa: BLE001
            print(f"[document_ocr] DirectML warm-up failed ({e}); falling back to CPU")
            self.using_dml = False

    def run(self, img_bgr: np.ndarray, lang: Optional[str] = None) -> OCRResult:
        engine = self._engine_for(lang)
        t0 = time.time()
        res = engine(img_bgr)
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
                               "n_tokens": len(tokens),
                               "lang": normalize_lang(lang)})

    def recognize_crop(self, crop_bgr: np.ndarray, lang: Optional[str] = None) -> Tuple[str, float]:
        """Recognition-only read of one crop.

        Upstream RapidOCR 3.x bug: `__call__` forwards use_det/use_cls to
        `update_params`, which setattr()s them on the engine *permanently* (no
        restore after the call). One rec-only call therefore disables detection
        for every later page — silently yielding zero tokens. We save/restore
        the flags around the call (verified by test_document_ocr_state.py).
        """
        engine = self._engine_for(lang)
        saved = (engine.use_det, engine.use_cls, engine.use_rec)
        try:
            out = engine(crop_bgr, use_det=False, use_cls=False, use_rec=True)
        finally:
            (engine.use_det, engine.use_cls, engine.use_rec) = saved
        txts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        if not txts:
            return "", 0.0
        return str(txts[0]), float(scores[0]) * 100.0 if len(scores) else 0.0

    def line_orientation_votes(self, img_bgr: np.ndarray,
                               result: OCRResult,
                               lang: Optional[str] = None) -> Optional[Tuple[float, float, int]]:
        """Re-run the PP-LCNet line classifier on the detected line crops.

        RapidOCR 3.x applies the classifier internally (rotating crops), but
        does not surface the labels; re-running it on our own crops is cheap
        and gives page-flip evidence the public output lacks. Returns
        (frac180, hi_conf_180, n_lines) - see the base-class docstring.

        Not available for non-default languages: the classifier is trained on
        Chinese/English and measurably flips Devanagari crops.
        """
        if normalize_lang(lang) != "default":
            return None
        if not result.tokens:
            return None
        self._ensure()
        h, w = img_bgr.shape[:2]
        crops: List[np.ndarray] = []
        for tok in result.tokens:
            x0, y0, x1, y1 = tok.bbox
            x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
            x1, y1 = min(w, x1 + 2), min(h, y1 + 2)
            if x1 > x0 and y1 > y0:
                crops.append(img_bgr[y0:y1, x0:x1])
        if not crops:
            return None
        try:
            out = self._engine.text_cls(crops)
        except Exception:  # noqa: BLE001
            return None
        labels = getattr(out, "cls_res", None) or []
        if not labels:
            return None
        n = len(labels)
        n180 = sum(1 for lab, _score in labels if str(lab) == "180")
        hi180 = sum(1 for lab, score in labels
                    if str(lab) == "180" and float(score) >= 0.9)
        return n180 / n, hi180 / n, n


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

    def recognize_crop(self, crop_bgr: np.ndarray, lang: Optional[str] = None) -> Tuple[str, float]:
        exe = self._find()
        if not exe:
            raise RuntimeError("tesseract binary not found")
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "crop.png")
            cv2.imwrite(inp, crop_bgr)
            cmd = [exe, inp, "stdout", "-l", lang or "eng", "--psm", "7", "tsv"]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=120)
            tokens = self._parse_tsv(proc.stdout or "")
        if not tokens:
            return "", 0.0
        text = " ".join(t.text for t in tokens)
        conf = float(np.mean([t.conf for t in tokens]))
        return text, conf


# ---------------------------------------------------------------------------
# registry + flagging helpers
# ---------------------------------------------------------------------------

_BACKENDS = {"rapidocr": RapidOCRBackend, "tesseract": TesseractBackend}
_INSTANCES: Dict[str, OCRBackend] = {}


def available_backends() -> List[str]:
    out = []
    for name, cls in _BACKENDS.items():
        ok, _why = cls().available()
        if ok:
            out.append(name)
    return out


def get_backend(name: str) -> OCRBackend:
    """Cached backend instance. Engine construction (ONNX sessions) is expensive;
    re-creating it per call was measurable overhead in batch runs."""
    if name not in _BACKENDS:
        raise ValueError(f"unknown OCR backend {name!r}; have {sorted(_BACKENDS)}")
    inst = _INSTANCES.get(name)
    if inst is None:
        inst = _BACKENDS[name]()
        _INSTANCES[name] = inst
    ok, why = inst.available()
    if not ok:
        raise RuntimeError(f"OCR backend {name!r} unavailable: {why}")
    return inst


def ocr_page(img_bgr: np.ndarray, backend: str = "rapidocr",
             lang: Optional[str] = None, conf_threshold: float = 60.0,
             recheck_digits: bool = False,
             repass_conf_below: float = 95.0) -> OCRResult:
    """Run OCR, attach flags, optionally re-read digit tokens on 2x crops.

    The re-pass never replaces text; it records `repass_text`/`repass_conf` and
    raises `digit_conflict` when the digit sequences disagree.

    `repass_conf_below=100` re-reads every digit token (PP-OCR confidences are
    overconfident: ECE ~0.33 on the frozen set), at a small latency cost.
    """
    be = get_backend(backend)
    result = be.run(img_bgr, lang=lang)
    devanagari = normalize_lang(lang) == "devanagari"
    calibration = None
    if devanagari:
        from calibration import load_calibration
        calibration = load_calibration()
        if calibration:
            result.meta["calibration"] = f"isotonic:{os.path.basename(calibration.get('domain', '?'))}"
    result.meta["flagged"] = flag_tokens(result.tokens, conf_threshold,
                                         devanagari=devanagari,
                                         calibration=calibration)
    result.meta["conf_threshold"] = conf_threshold
    if recheck_digits:
        conflicts = apply_digit_repass(result.tokens, img_bgr, be.recognize_crop,
                                       lang=lang, conf_below=repass_conf_below)
        result.meta["digit_repass_conflicts"] = conflicts
        result.meta["repass_conf_below"] = repass_conf_below
    result.meta["flagged"] = sum(1 for t in result.tokens if t.flags)
    return result

# ---------------------------------------------------------------------------
# digit re-pass (recognition-only second look, never a silent replacement)
# ---------------------------------------------------------------------------

def _crop_with_pad(img: np.ndarray, bbox: Tuple[int, int, int, int],
                   pad_ratio: float = 0.15) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    px, py = int(w * pad_ratio) + 2, int(h * pad_ratio) + 2
    xa, ya = max(0, x0 - px), max(0, y0 - py)
    xb, yb = min(img.shape[1], x1 + px), min(img.shape[0], y1 + py)
    return img[ya:yb, xa:xb].copy()


def apply_digit_repass(tokens: List[Token], img_bgr: np.ndarray,
                       repass_fn, lang: Optional[str] = None,
                       scale: float = 2.0, conf_below: float = 95.0,
                       max_tokens: int = 40) -> int:
    """Re-read digit tokens on upscaled crops. Returns conflict count.

    Agreement is recorded in `repass_text` only (no flag) so that flag-based
    coverage metrics are not polluted by successful checks.
    """
    checked = conflicts = 0
    for tok in tokens:
        if not tok.has_digits or tok.conf >= conf_below:
            continue
        if checked >= max_tokens:
            break
        crop = _crop_with_pad(img_bgr, tok.bbox)
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        try:
            text, conf = repass_fn(big, lang) if lang else repass_fn(big)
        except TypeError:
            text, conf = repass_fn(big)
        except Exception:
            continue
        tok.repass_text, tok.repass_conf = text, conf
        checked += 1
        if _digits_of(tok.text) != _digits_of(text):
            if "digit_conflict" not in tok.flags:
                tok.flags.append("digit_conflict")
            if tok.alt_text is None:
                tok.alt_text = text
            conflicts += 1
    return conflicts


# ---------------------------------------------------------------------------
# do-not-hallucinate gate: two streams must agree on digits
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# do-not-hallucinate gate: two streams must agree on digits
# ---------------------------------------------------------------------------

RISK_WEIGHTS: Dict[str, float] = {
    "digit_conflict": 3.0,   # two independent streams read different digits
    "cross_model_conflict": 3.0,  # a second model read different digits (Appendix L/M)
    "script_mismatch": 2.5,  # Latin token on a Devanagari page: measured ~100% junk
    "invalid_sequence": 1.0,  # impossible Devanagari sequence; dev-tuned to
                              # not displace digit signals in the top-10
    "digit_uncertain": 1.5,  # digit token below the digit-confidence bar
    "low_conf": 1.0,
}

_LATIN_RE = re.compile(r"[A-Za-z]")


def flag_tokens(tokens: List["Token"], conf_threshold: float,
                devanagari: bool = False,
                calibration: Optional[Dict] = None,
                deva_digit_conf: float = 90.0) -> int:
    """Attach honesty flags (never changes text). Returns flagged count.

    Devanagari-specific signals measured on the dev set (Appendix M):
    * Latin letters on a Devanagari page are ~100% junk (watermarks, garbage
      letter substitutions) -> `script_mismatch`;
    * the mobile recognizer's digit confidence bar needs 90, not 80: at 80 it
      caught only 23% of digit errors, at 90 it catches 69%.
    """
    for tok in tokens:
        if tok.conf < conf_threshold:
            tok.flags.append("low_conf")
        digit_bar = deva_digit_conf if devanagari else 80.0
        if tok.has_digits and tok.conf < digit_bar:
            tok.flags.append("digit_uncertain")
        if devanagari and _LATIN_RE.search(tok.text):
            tok.flags.append("script_mismatch")
        if devanagari:
            # Hypothesis-side validity: OCR output containing an impossible
            # combining sequence (reordered matra, dangling virama) is wrong by
            # construction - a near-free, high-precision queue signal (W0.1).
            from doc_metrics import _invalid_devanagari_token
            if _invalid_devanagari_token(tok.text):
                tok.flags.append("invalid_sequence")
        if calibration is not None:
            from calibration import apply_isotonic
            tok.cal_conf = round(apply_isotonic(tok.conf, calibration["isotonic"]), 2)
    return sum(1 for t in tokens if t.flags)


def apply_digit_verifier(tokens: List["Token"], img_bgr: np.ndarray,
                         verify_fn, pad_ratio: float = 0.08,
                         max_tokens: int = 12,
                         only_flagged: bool = True) -> int:
    """Second-model digit re-read on suspect tokens. Returns conflict count.

    `verify_fn(crops) -> texts` may be batch or single-crop (both accepted).
    Never replaces text: a disagreeing read raises `cross_model_conflict` and
    is stored as `alt_text` (same contract as the dual-stream conflict).
    On the frozen letterpress set this flag had recall 0.71 / precision 0.81
    against baseline digit errors (Appendix L, bodhan verifier role).
    """
    suspects = [t for t in tokens
                if t.has_digits and (not only_flagged or t.flags)]
    if max_tokens:
        suspects = suspects[:max_tokens]
    if not suspects:
        return 0
    crops = [_crop_with_pad(img_bgr, t.bbox, pad_ratio) for t in suspects]
    try:
        texts = verify_fn(crops)
    except TypeError:
        texts = [verify_fn(c) for c in crops]
    conflicts = 0
    for tok, text in zip(suspects, texts):
        if text is None:
            continue
        if _digits_of(str(text)) and _digits_of(str(text)) != _digits_of(tok.text):
            tok.flags.append("cross_model_conflict")
            tok.alt_text = str(text).strip()
            conflicts += 1
    return conflicts


def token_risk(tok: "Token") -> float:
    """Uncertainty score used to rank the human review queue.

    Evidence-based on the SROIE/CORD sweeps: single signals do not meet
    precision bars on real photos, but ranking multiplies their value -
    'check these 5 numbers first' is what a human can act on.
    """
    risk = sum(RISK_WEIGHTS.get(f, 1.0) for f in tok.flags)
    if tok.repass_text is not None and _digits_of(tok.repass_text) != _digits_of(tok.text):
        risk += 2.0  # a recorded 2x re-read disagrees
    if tok.alt_text is not None:
        risk += 0.5  # an alternative reading was recorded at all
    return risk


def review_queue(tokens: List["Token"], top_k: Optional[int] = None) -> List["Token"]:
    """Tokens flagged for human review, riskiest first (stable reading order)."""
    risky = [t for t in tokens if t.flags]
    ranked = sorted(risky, key=lambda t: (-token_risk(t), t.bbox[1], t.bbox[0]))
    return ranked[:top_k] if top_k else ranked


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
                  conf_threshold: float = 60.0,
                  recheck_digits: bool = False) -> OCRResult:
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
                       conf_threshold=conf_threshold, recheck_digits=recheck_digits)
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
