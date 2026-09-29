"""Run the document pipeline for one uploaded page (or a whole PDF).

The web studio's document entry point: canonical file names, a curated meta
block, the review queue, a flags summary, and the transcript. PDFs default to
the first page; with ``all_pages`` every page is processed (capped) and a
combined PDF/TXT/MD is written next to the page-1 artifacts.
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

#: PDF pages processed per run when ``all_pages`` is on (keeps the single
#: heavy worker bounded; the response reports what was processed).
PAGE_CAP = 10

CANONICAL = {
    "restored.png": "image/png",
    "overlay.png": "image/png",
    "searchable.pdf": "application/pdf",
    "transcript.txt": "text/plain; charset=utf-8",
    "transcript.md": "text/markdown; charset=utf-8",
    "combined.pdf": "application/pdf",
    "combined.txt": "text/plain; charset=utf-8",
    "combined.md": "text/markdown; charset=utf-8",
    "ocr.json": "application/json",
}


class PipelineError(RuntimeError):
    pass


def _load_pages(path: str, kind: Optional[str],
                all_pages: bool) -> Tuple[List[Tuple[int, "object"]], str]:
    """Return ([(page_index, BGR page)], kind).

    `kind` ("image" | "pdf", from the upload validator) wins over the file
    extension: streamed uploads land on disk without one. PDFs default to the
    first page; ``all_pages`` expands to (at most PAGE_CAP) pages.
    """
    is_pdf = kind == "pdf" if kind else path.lower().endswith(".pdf")
    if is_pdf:
        import doc_data
        pages: List[Tuple[int, "object"]] = []
        for idx, img_bgr, _gt in doc_data.pdf_to_pages(path, dpi=200):
            pages.append((idx, img_bgr))
            if not all_pages or len(pages) >= PAGE_CAP:
                break
        if not pages:
            raise PipelineError("Could not read that PDF.")
        return pages, "pdf"

    from document_orientation import load_image_bgr
    img_bgr = load_image_bgr(path)
    if img_bgr is None:
        raise PipelineError("Could not read that image.")
    return [(0, img_bgr)], "image"


def _copy_or_text(run_dir: str, name: str, src: Optional[str],
                  fallback_text: Optional[str] = None) -> Optional[str]:
    """Move a pipeline output to its canonical name; returns the final path."""
    dst = os.path.join(run_dir, name)
    if src and os.path.isfile(src):
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
        return dst
    if fallback_text is not None:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(fallback_text)
        return dst
    return None


def process_page(upload_path: str, run_dir: str, run_id: str, *,
                 lang: Optional[str], deskew: bool,
                 kind: Optional[str] = None, ocr: Optional[str] = None,
                 make_overlay: bool = True, make_pdf: bool = True,
                 make_txt: bool = True, make_md: bool = True,
                 all_pages: bool = False,
                 auto_rotate: bool = True) -> Dict:
    """Run restore+OCR and write canonical artifacts into run_dir."""
    import cv2

    from document_export import (write_combined_transcript_md,
                                 write_searchable_pdf_pages)
    from document_pipeline import run_document_pipeline

    started = time.time()
    pages, kind = _load_pages(upload_path, kind, all_pages)
    backend = None if ocr in (None, "auto") else ocr

    stem = f"page_{run_id[:8]}"
    results = []
    for idx, img_bgr in pages:
        page_stem = stem if idx == pages[0][0] else f"{stem}_p{idx:03d}"
        results.append((page_stem, run_document_pipeline(
            img_bgr, backend=backend, deskew=deskew, lang=lang,
            auto_rotate=auto_rotate, out_dir=run_dir, stem=page_stem,
            make_pdf=make_pdf, make_overlay=make_overlay,
            make_txt=make_txt, make_json=True, make_md=make_md)))

    _, result = results[0]

    # Canonical page-1 artifacts -------------------------------------------
    cv2.imwrite(os.path.join(run_dir, "restored.png"), result.display_bgr)
    outputs = getattr(result, "outputs", {}) or {}
    files: Dict[str, Optional[str]] = {
        "overlay.png": _copy_or_text(run_dir, "overlay.png",
                                     outputs.get("overlay")),
        "searchable.pdf": _copy_or_text(run_dir, "searchable.pdf",
                                        outputs.get("pdf")),
        "transcript.txt": _copy_or_text(
            run_dir, "transcript.txt", outputs.get("txt"),
            fallback_text=(result.ocr.text if getattr(result, "ocr", None) else "")
            if make_txt else None),
        "transcript.md": _copy_or_text(run_dir, "transcript.md",
                                       outputs.get("md")),
        "ocr.json": _copy_or_text(run_dir, "ocr.json", outputs.get("json")),
        "restored.png": os.path.join(run_dir, "restored.png"),
    }

    # Multi-page combined outputs ------------------------------------------
    if len(results) > 1:
        write_searchable_pdf_pages(
            os.path.join(run_dir, "combined.pdf"),
            [(r.display_bgr, r.ocr.tokens, 200) for _s, r in results],
            title=stem)
        with open(os.path.join(run_dir, "combined.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n\n".join(r.ocr.text for _s, r in results) + "\n")
        if make_md:
            write_combined_transcript_md(
                os.path.join(run_dir, "combined.md"),
                [r.ocr for _s, r in results], title=stem)
        files["combined.pdf"] = os.path.join(run_dir, "combined.pdf")
        files["combined.txt"] = os.path.join(run_dir, "combined.txt")
        if make_md:
            files["combined.md"] = os.path.join(run_dir, "combined.md")

    # Review queue + summary from the written ocr.json (single source) -----
    review: List[Dict] = []
    summary: Dict[str, int] = {}
    n_tokens = 0
    try:
        with open(os.path.join(run_dir, "ocr.json"), encoding="utf-8") as f:
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
        with open(os.path.join(run_dir, "transcript.txt"),
                  encoding="utf-8") as f:
            transcript = f.read()
    except OSError:
        pass

    meta_src = dict(getattr(result, "meta", {}) or {})
    meta = {
        "seconds": round(float(meta_src.get("seconds", time.time() - started)), 2),
        "backend": meta_src.get("backend", backend or "rapidocr"),
        "primary_stream": meta_src.get("primary_stream"),
        "skew_angle": meta_src.get("skew_angle", 0.0),
        "resized": bool(meta_src.get("resized", False)),
        "n_tokens": n_tokens,
        "flagged": len(review),
        "digit_conflicts": summary.get("digit_conflict", 0),
        "lang": lang or "default",
        "input_kind": kind,
        "deskew": deskew,
        "auto_rotate": auto_rotate,
        "all_pages": all_pages,
        "pages_processed": len(results),
        "pages_capped": bool(all_pages and len(results) >= PAGE_CAP),
        "page_cap": PAGE_CAP,
    }

    key_map = {
        "restored.png": "restored",
        "overlay.png": "overlay",
        "searchable.pdf": "pdf",
        "transcript.txt": "txt",
        "transcript.md": "md",
        "ocr.json": "json",
        "combined.pdf": "combined_pdf",
        "combined.txt": "combined_txt",
        "combined.md": "combined_md",
    }
    return {
        "run_id": run_id,
        "meta": meta,
        "flags_summary": summary,
        "review": review,
        "transcript": transcript,
        "status_line": getattr(result, "status_line", ""),
        "files": {
            key_map[key]: f"/api/runs/{run_id}/files/{key}"
            for key, path in files.items() if path and key in key_map
        },
    }
