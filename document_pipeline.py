"""
document_pipeline.py — the single document-restore pipeline used by CLI and app.

    restore -> dual-stream OCR (primary per RECOMMENDED_STREAM, auditor the other)
            -> digit-conflict flags (never picks a winner)
            -> outputs (searchable PDF, overlay, transcript, OCR JSON)

API:
    run_document_pipeline(img_bgr, *, backend=None, lang=None, deskew=False,
                          repass_digits=False, dpi=None, out_dir=None, stem="page",
                          make_pdf=True, make_overlay=True, make_txt=True,
                          make_json=True) -> DocumentResult

`DocumentResult.display_bgr` is the cleaned page embedded in the PDF/overlay;
`DocumentResult.ocr.tokens` bboxes live in display coordinates by construction:
  * deskew=False -> display keeps raw geometry; primary stream per
    RECOMMENDED_STREAM (rapidocr: raw, tesseract: restored)
  * deskew=True  -> display is rotated, so primary OCR runs on the display
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from document_ocr import (RECOMMENDED_STREAM, OCRResult, apply_digit_verifier,
                          available_backends, compare_digit_streams, get_backend,
                          ocr_page, review_queue)
from document_orientation import (VERTICAL_ACTION, VOTE180_FRAC, VOTE180_HI,
                                  RotationInfo, detect_rotation, infer_angle,
                                  rotate_bgr, vertical_fraction)
from document_restore import restore_document


@dataclass
class DocumentResult:
    display_bgr: np.ndarray
    ocr: OCRResult
    meta: Dict = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)

    @property
    def review_list(self):
        """Flagged tokens, riskiest first - the human review queue."""
        return review_queue(self.ocr.tokens)

    @property
    def status_line(self) -> str:
        low = sum(1 for t in self.ocr.tokens if "low_conf" in t.flags)
        conflicts = self.meta.get("digit_conflicts", 0)
        review = len(self.review_list)
        orient = " · orientation?" if self.meta.get("orientation_suspect") else ""
        return (f"{len(self.ocr.tokens)} tokens · {low} low-confidence · "
                f"{conflicts} digit conflicts · {review} to review{orient} · "
                f"{self.meta.get('seconds', 0):.1f}s")


def pick_backend(backend: Optional[str] = None) -> str:
    avail = available_backends()
    if not avail:
        raise RuntimeError("no OCR backend available (pip install rapidocr)")
    if backend:
        if backend not in avail:
            raise RuntimeError(f"OCR backend {backend!r} unavailable; have {avail}")
        return backend
    return "rapidocr" if "rapidocr" in avail else avail[0]


@dataclass
class _Pass:
    """One full restore + dual-stream OCR pass (orientation may re-run it)."""
    restored: Dict
    display: np.ndarray
    primary_img: np.ndarray
    primary_name: str
    result: OCRResult
    conflicts: int
    audit_error: Optional[str]


def run_document_pipeline(img_bgr: np.ndarray, *, backend: Optional[str] = None,
                          lang: Optional[str] = None, deskew: bool = False,
                          scale: int = 1, repass_digits: bool = False,
                          reading_order: bool = True,
                          auto_rotate: bool = True,
                          primary_stream: Optional[str] = None,
                          digit_verifier=None,
                          verifier_scope: str = "flagged",
                          dpi: Optional[int] = None,
                          out_dir: Optional[str] = None, stem: str = "page",
                          make_pdf: bool = True, make_overlay: bool = True,
                          make_txt: bool = True, make_json: bool = True,
                          conf_threshold: float = 60.0) -> DocumentResult:
    """`primary_stream` ("raw"|"restored", None = RECOMMENDED_STREAM) exists
    for evaluation A/Bs; production callers leave it unset.

    `digit_verifier` is an optional callable (crops -> texts) that re-reads
    suspect digit tokens; disagreements become `cross_model_conflict` flags
    with the alternative reading in `alt_text`. Flag-only; see
    document_verifier.py. `verifier_scope` is "flagged" (default: only tokens
    the first pass already suspects) or "all" (every digit token, costlier)."""
    t0 = time.time()
    backend = pick_backend(backend)
    be = get_backend(backend)

    def _run_pass(image: np.ndarray) -> _Pass:
        restored = restore_document(image, scale=scale, deskew=deskew)
        display = restored["display_bgr"]

        # Primary stream follows the measured table (RECOMMENDED_STREAM); the
        # other stream audits digit tokens. deskew=True forces the display.
        stream = primary_stream or RECOMMENDED_STREAM.get(backend, "raw")
        if deskew:
            primary_img, audit_img, primary_name = display, image, "display"
        elif stream.startswith("restore"):
            primary_img, audit_img, primary_name = display, image, "restored"
        else:
            primary_img, audit_img, primary_name = image, display, "raw"

        result = ocr_page(primary_img, backend=backend, lang=lang,
                          conf_threshold=conf_threshold, recheck_digits=repass_digits)
        conflicts = 0
        audit_error = None
        try:
            audit = ocr_page(audit_img, backend=backend, lang=lang,
                             conf_threshold=conf_threshold)
            conflicts = compare_digit_streams(result, audit)
        except Exception as e:  # noqa: BLE001
            audit_error = str(e)
        return _Pass(restored, display, primary_img, primary_name, result,
                     conflicts, audit_error)

    # Pass 1 doubles as the orientation-evidence pass: the line classifier
    # votes and token geometry are read off its tokens (no extra OCR on the
    # common upright case). A rotation re-runs the pass on the rotated page.
    p = _run_pass(img_bgr)
    passes = 1
    votes = be.line_orientation_votes(p.primary_img, p.result, lang=lang)
    vfrac = vertical_fraction(p.result.tokens)
    info = infer_angle(p.result.tokens, votes, auto_rotate=auto_rotate)
    rot = info
    if info.source == "votes-180":
        p = _run_pass(rotate_bgr(img_bgr, 180))
        passes = 2
    elif info.source == "probe-needed" and backend == "rapidocr":
        # Sideways page: re-read it rotated 90° clockwise. If the probe is
        # upright, its votes pick 90 (low) vs 270 (high); if the probe is
        # still vertical, fall back to OSD and otherwise flag for review.
        probe = rotate_bgr(img_bgr, 90)
        pp = _run_pass(probe)
        passes = 2
        pvotes = be.line_orientation_votes(pp.primary_img, pp.result, lang=lang)
        if vertical_fraction(pp.result.tokens) < VERTICAL_ACTION:
            p = pp
            if pvotes is None:
                # No line classifier for this language (devanagari): the probe
                # fixed the axis, let OSD pick the direction instead of a coin
                # flip; without OSD the 90cw default stays and the suspect
                # flag covers it.
                osd = _osd_sideways(img_bgr)
                angle = osd.angle if osd.angle in (90, 270) else 90
                rot = RotationInfo(angle, osd.confidence, "probe-osd", raw_angle=angle)
            elif pvotes[0] >= VOTE180_FRAC and pvotes[1] >= VOTE180_HI:
                p = _run_pass(rotate_bgr(probe, 180))
                passes = 3
                rot = RotationInfo(270, round(pvotes[0], 3), "probe-270", raw_angle=270)
            else:
                conf = round(pvotes[0], 3) if pvotes else 0.0
                rot = RotationInfo(90, conf, "probe-90", raw_angle=90)
        else:
            rot = _osd_sideways(img_bgr)
            if rot.angle:
                p = _run_pass(rotate_bgr(img_bgr, rot.angle))
                passes = 2
    elif info.source == "probe-needed":
        rot = _osd_sideways(img_bgr)
        if rot.angle:
            p = _run_pass(rotate_bgr(img_bgr, rot.angle))
            passes = 2

    result = p.result
    result.meta["digit_conflicts"] = p.conflicts
    if digit_verifier is not None:
        try:
            vconflicts = apply_digit_verifier(result.tokens, p.primary_img,
                                              digit_verifier,
                                              only_flagged=verifier_scope != "all")
            result.meta["digit_verifier"] = {
                "name": getattr(digit_verifier, "name", "?"),
                "scope": verifier_scope,
                "conflicts": vconflicts,
            }
        except Exception as e:  # noqa: BLE001
            result.meta["digit_verifier_error"] = str(e)
    result.meta["primary_stream"] = p.primary_name
    if p.audit_error:
        result.meta["audit_error"] = p.audit_error
    result.meta["skew_angle"] = p.restored["debug"]["skew_angle"]

    if reading_order:
        from document_layout import sort_reading_order, text_in_order
        before = [t.text for t in result.tokens]
        stats: Dict = {}
        result.tokens = sort_reading_order(result.tokens, stats=stats)
        result.text = text_in_order(result.tokens)
        result.meta["reading_order_splits"] = stats.get("splits", 0)
        result.meta["reading_order_changed"] = \
            [t.text for t in result.tokens] != before
    result.meta["reading_order"] = reading_order

    vfinal = vertical_fraction(result.tokens)
    suspect = bool(
        vfinal >= VERTICAL_ACTION
        or rot.source in ("inconclusive", "osd-180", "vote-ambiguous")
        or (not auto_rotate and info.raw_angle in (90, 180))
    )
    result.meta["orientation_suspect"] = suspect
    result.meta["auto_rotate"] = rot.as_dict()
    result.meta["orientation_evidence"] = {
        "vote180": round(votes[0], 3) if votes else None,
        "vote180_hi": round(votes[1], 3) if votes else None,
        "n_votes": votes[2] if votes else 0,
        "vertical": round(vfrac, 3),
        "passes": passes,
    }

    outputs: Dict[str, str] = {}
    if out_dir:
        from document_export import export_document_outputs
        outputs = export_document_outputs(
            out_dir, stem, p.display, result, dpi=dpi,
            make_pdf=make_pdf, make_overlay=make_overlay, make_txt=make_txt,
            make_json=make_json, overlay_source=p.display)

    meta = {
        "seconds": round(time.time() - t0, 2),
        "backend": backend,
        "skew_angle": p.restored["debug"]["skew_angle"],
        "primary_stream": result.meta["primary_stream"],
        "digit_conflicts": p.conflicts,
        "reading_order": reading_order,
        "reading_order_splits": result.meta.get("reading_order_splits", 0),
        "reading_order_changed": result.meta.get("reading_order_changed", False),
        "orientation_suspect": suspect,
        "auto_rotate": rot.as_dict(),
        "orientation_evidence": result.meta["orientation_evidence"],
        "repass_digits": repass_digits,
    }
    return DocumentResult(display_bgr=p.display, ocr=result, meta=meta, outputs=outputs)


def _osd_sideways(img_bgr: np.ndarray) -> RotationInfo:
    """OSD fallback for 90/270 when OCR evidence is inconclusive/unavailable."""
    osd = detect_rotation(img_bgr)
    if osd.angle in (90, 270):
        return RotationInfo(osd.angle, osd.confidence, "osd-sideways",
                            raw_angle=osd.angle)
    if osd.source == "osd-180":
        return osd
    return RotationInfo(0, 0.0, "inconclusive")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import doc_data
    from degradation_document import degrade_page

    page, _gt = doc_data.render_synthetic_invoice(seed=61, dpi=200)
    degraded = degrade_page(page, seed=6101, level="medium")
    res = run_document_pipeline(degraded, out_dir="out/api_check/pipeline_demo", stem="invoice")
    print("status:", res.status_line)
    print("outputs:", {k: os.path.basename(v) for k, v in res.outputs.items()})
    print("document_pipeline OK")
