"""Run the document pipeline for one uploaded page.

The web studio's document entry point: canonical file names, a curated meta
block, the review queue, a flags summary, and the transcript.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys  # noqa: E402

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CANONICAL = {
    "restored.png": "image/png",
    "overlay.png": "image/png",
    "searchable.pdf": "application/pdf",
    "transcript.txt": "text/plain; charset=utf-8",
    "transcript.md": "text/markdown; charset=utf-8",
    "ocr.json": "application/json",
}


class PipelineError(RuntimeError):
    pass


def _load_page(path: str, kind: Optional[str] = None) -> "object":
    """Load an uploaded file as a BGR page (image with EXIF applied, or PDF
    first page at 200 dpi). Returns (bgr, kind).

    `kind` ("image" | "pdf", from the upload validator) wins over the file
    extension: streamed uploads land on disk without one.
    """
    import cv2
    import numpy as np

    is_pdf = kind == "pdf" if kind else path.lower().endswith(".pdf")
    if is_pdf:
        import doc_data
        pages = list(doc_data.pdf_to_pages(path, dpi=200))
        if not pages:
            raise PipelineError("Could not read that PDF.")
        _idx, img_bgr, _gt = pages[0]
        return img_bgr, "pdf"

    from document_orientation import load_image_bgr
    img_bgr = load_image_bgr(path)
    if img_bgr is None:
        raise PipelineError("Could not read that image.")
    return img_bgr, "image"


def process_page(upload_path: str, run_dir: str, run_id: str, *,
                 lang: Optional[str], deskew: bool,
                 kind: Optional[str] = None) -> Dict:
    """Run restore+OCR on one page and write canonical artifacts into run_dir."""
    import cv2

    from document_pipeline import run_document_pipeline

    started = time.time()
    img_bgr, kind = _load_page(upload_path, kind)

    stem = f"page_{run_id[:8]}"
    result = run_document_pipeline(
        img_bgr, backend=None, deskew=deskew, lang=lang,
        out_dir=run_dir, stem=stem,
        make_pdf=True, make_overlay=True, make_txt=True, make_json=True,
        make_md=True,
    )

    # Canonical artifacts -------------------------------------------------
    restored_path = os.path.join(run_dir, "restored.png")
    cv2.imwrite(restored_path, result.display_bgr)

    outputs = getattr(result, "outputs", {}) or {}
    copies = {
        "overlay.png": outputs.get("overlay"),
        "searchable.pdf": outputs.get("pdf"),
        "transcript.txt": outputs.get("txt"),
        "transcript.md": outputs.get("md"),
        "ocr.json": outputs.get("json"),
    }
    for name, src in copies.items():
        dst = os.path.join(run_dir, name)
        if src and os.path.isfile(src):
            if os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)
        elif name == "transcript.txt":
            with open(dst, "w", encoding="utf-8") as f:
                f.write(result.ocr.text if getattr(result, "ocr", None) else "")

    # Review queue + summary from the written ocr.json (single source) ----
    review: List[Dict] = []
    summary: Dict[str, int] = {}
    n_tokens = 0
    ocr_json_path = os.path.join(run_dir, "ocr.json")
    try:
        with open(ocr_json_path, encoding="utf-8") as f:
            payload = json.load(f)
        n_tokens = len(payload.get("tokens", []))
        for item in payload.get("review", []):
            entry = {
                "text": item.get("text", ""),
                "alt_text": item.get("alt_text"),
                "flags": item.get("flags", []),
                "conf": item.get("conf"),
                "risk": item.get("risk"),
                "bbox": item.get("bbox"),
            }
            review.append(entry)
            for flag in entry["flags"]:
                summary[flag] = summary.get(flag, 0) + 1
    except (OSError, ValueError):
        pass

    transcript = ""
    try:
        with open(os.path.join(run_dir, "transcript.txt"), encoding="utf-8") as f:
            transcript = f.read()
    except OSError:
        pass

    meta_src = dict(getattr(result, "meta", {}) or {})
    meta = {
        "seconds": round(float(meta_src.get("seconds", time.time() - started)), 2),
        "backend": meta_src.get("backend", "rapidocr"),
        "primary_stream": meta_src.get("primary_stream"),
        "skew_angle": meta_src.get("skew_angle", 0.0),
        "resized": bool(meta_src.get("resized", False)),
        "n_tokens": n_tokens,
        "flagged": len(review),
        "digit_conflicts": summary.get("digit_conflict", 0),
        "lang": lang or "default",
        "input_kind": kind,
        "deskew": deskew,
    }

    return {
        "run_id": run_id,
        "meta": meta,
        "flags_summary": summary,
        "review": review,
        "transcript": transcript,
        "status_line": getattr(result, "status_line", ""),
        "files": {
            "restored": f"/api/runs/{run_id}/files/restored.png",
            "overlay": f"/api/runs/{run_id}/files/overlay.png",
            "pdf": f"/api/runs/{run_id}/files/searchable.pdf",
            "txt": f"/api/runs/{run_id}/files/transcript.txt",
            "md": f"/api/runs/{run_id}/files/transcript.md",
            "json": f"/api/runs/{run_id}/files/ocr.json",
        },
    }
