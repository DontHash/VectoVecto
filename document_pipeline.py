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

from document_ocr import (RECOMMENDED_STREAM, OCRResult, available_backends,
                          compare_digit_streams, ocr_page, review_queue)
from document_orientation import (RotationInfo, detect_rotation,
                                  orientation_suspect, rotate_bgr)
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


def run_document_pipeline(img_bgr: np.ndarray, *, backend: Optional[str] = None,
                          lang: Optional[str] = None, deskew: bool = False,
                          scale: int = 1, repass_digits: bool = False,
                          reading_order: bool = True,
                          auto_rotate: bool = True,
                          dpi: Optional[int] = None,
                          out_dir: Optional[str] = None, stem: str = "page",
                          make_pdf: bool = True, make_overlay: bool = True,
                          make_txt: bool = True, make_json: bool = True,
                          conf_threshold: float = 60.0) -> DocumentResult:
    t0 = time.time()
    backend = pick_backend(backend)

    rot = RotationInfo()
    if auto_rotate:
        rot = detect_rotation(img_bgr)
        if rot.angle:
            img_bgr = rotate_bgr(img_bgr, rot.angle)

    restored = restore_document(img_bgr, scale=scale, deskew=deskew)
    display = restored["display_bgr"]

    # Primary stream follows the measured table (RECOMMENDED_STREAM); the other
    # stream audits digit tokens. deskew=True forces the display (raw is rotated).
    stream = RECOMMENDED_STREAM.get(backend, "raw")
    if deskew:
        primary_img, audit_img = display, img_bgr
        primary_name = "display"
    elif stream.startswith("restore"):
        primary_img, audit_img = display, img_bgr
        primary_name = "restored"
    else:
        primary_img, audit_img = img_bgr, display
        primary_name = "raw"

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

    result.meta["digit_conflicts"] = conflicts
    result.meta["primary_stream"] = primary_name
    if audit_error:
        result.meta["audit_error"] = audit_error
    result.meta["skew_angle"] = restored["debug"]["skew_angle"]

    if reading_order:
        from document_layout import sort_reading_order, text_in_order
        result.tokens = sort_reading_order(result.tokens)
        result.text = text_in_order(result.tokens)
    result.meta["reading_order"] = reading_order
    suspect = orientation_suspect(result.tokens) or rot.source == "osd-180"
    result.meta["orientation_suspect"] = suspect
    result.meta["auto_rotate"] = rot.as_dict()

    outputs: Dict[str, str] = {}
    if out_dir:
        from document_export import export_document_outputs
        outputs = export_document_outputs(
            out_dir, stem, display, result, dpi=dpi,
            make_pdf=make_pdf, make_overlay=make_overlay, make_txt=make_txt,
            make_json=make_json, overlay_source=display)

    meta = {
        "seconds": round(time.time() - t0, 2),
        "backend": backend,
        "skew_angle": restored["debug"]["skew_angle"],
        "primary_stream": result.meta["primary_stream"],
        "digit_conflicts": conflicts,
        "reading_order": reading_order,
        "orientation_suspect": suspect,
        "auto_rotate": rot.as_dict(),
        "repass_digits": repass_digits,
    }
    return DocumentResult(display_bgr=display, ocr=result, meta=meta, outputs=outputs)


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
