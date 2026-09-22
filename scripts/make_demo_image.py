"""
make_demo_image.py — build the README before/after figure from the real pipeline.

Two rows, each: raw input -> pipeline output (restored page + numbered review
boxes). Row 1 is a letterpress book page from heiDATA (CC BY 4.0,
doi:10.11588/data/EGOKEI); row 2 is this repository's own synthetic invoice
with heavy phone-photo degradation. No internal evaluation data is used.

    python scripts/make_demo_image.py
    # writes docs/assets/before_after.png
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
from degradation_document import degrade_page  # noqa: E402
from document_export import write_overlay_png  # noqa: E402
from document_pipeline import run_document_pipeline  # noqa: E402

PANEL_H = 760
GAP = 26
PAD = 24
LABEL_H = 40
OUT = os.path.join(BASE_DIR, "docs", "assets", "before_after.png")


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h),
                      interpolation=cv2.INTER_AREA)


def _labeled(img: np.ndarray, label: str, color=(30, 30, 30)) -> np.ndarray:
    """Panel with a caption strip on top."""
    panel = np.full((img.shape[0] + LABEL_H, img.shape[1], 3), 255, np.uint8)
    panel[LABEL_H:] = img
    cv2.putText(panel, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                color, 2, cv2.LINE_AA)
    return panel


def _arrow(h: int) -> np.ndarray:
    """Arrow strip matching the labelled panelt height."""
    w = 110
    canvas = np.full((h + LABEL_H, w, 3), 255, np.uint8)
    y = (h + LABEL_H) // 2
    cv2.arrowedLine(canvas, (12, y), (w - 18, y), (90, 90, 90), 4,
                    cv2.LINE_AA, tipLength=0.32)
    return canvas


def _pipeline_pair(img: np.ndarray, lang: str, stem: str,
                   deva_lines: str = "off") -> Tuple[np.ndarray, np.ndarray]:
    """(raw panel, overlay panel) for one page through the real pipeline."""
    tmp = tempfile.mkdtemp(prefix="vv_demo_")
    try:
        res = run_document_pipeline(
            img, backend="rapidocr", lang=lang, out_dir=tmp, stem=stem,
            make_pdf=False, make_overlay=False, make_txt=False, make_json=False,
            deva_lines=deva_lines)
        overlay_path = os.path.join(tmp, f"{stem}_overlay.png")
        write_overlay_png(overlay_path, res.display_bgr, res.ocr.tokens)
        return img, doc_data.imread_safe(overlay_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _letterpress_pair() -> Tuple[np.ndarray, np.ndarray]:
    """A heiDATA letterpress page (CC BY 4.0) with a mixed review queue.

    jagannatha1955_p03 carries digit conflicts, uncertain digits and invalid
    sequences, so the figure shows green/amber/red boxes, not just green.
    """
    manifest = doc_data.load_dataset(os.path.join(BASE_DIR, "data", "doc_eval",
                                                  "heidata_printed"))
    entry = next(e for e in manifest["entries"]
                 if e["id"].endswith("jagannatha1955_p03"))
    img = doc_data.imread_safe(entry["_degraded_path"])
    return _pipeline_pair(img, "ne", "letterpress")


def _photo_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Our own synthetic invoice as a phone photo: shadow + blur + noise.

    A gradient shadow is the artefact the restore stream removes best, so the
    before/after shows the product's actual job (not just noise).
    """
    page, _gt = doc_data.render_devanagari_invoice(seed=7, dpi=200)
    h, w = page.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.25 - 0.55 * (xx / w) - 0.25 * (yy / h)  # bright corner -> dark
    photo = np.clip(page.astype(np.float32) * shade[:, :, None], 0, 255)
    rng = np.random.default_rng(11)
    photo = cv2.GaussianBlur(photo, (0, 0), 1.1)
    photo = np.clip(photo + rng.normal(0, 4.0, photo.shape), 0, 255)
    return _pipeline_pair(photo.astype(np.uint8), "ne", "photo")


def build() -> str:
    rows: List[Tuple[str, Tuple[np.ndarray, np.ndarray]]] = [
        ("he? letterpress book scan (CC BY 4.0, heiDATA)", _letterpress_pair()),
        ("phone photo of a printed invoice (synthetic fixture)", _photo_pair()),
    ]
    panels: List[List[np.ndarray]] = []
    for label, (raw, overlay) in rows:
        raw_s = _resize_h(raw, PANEL_H)
        ov_s = _resize_h(overlay, PANEL_H)
        if ov_s.shape[1] != raw_s.shape[1]:  # identical input size
            ov_s = cv2.resize(overlay, (raw_s.shape[1], PANEL_H),
                              interpolation=cv2.INTER_AREA)
        letterpress = "letterpress" in label
        left = _labeled(raw_s, "INPUT" if letterpress else "INPUT (photo)")
        right = _labeled(ov_s, "OUTPUT: restored + review queue")
        panels.append([left, _arrow(PANEL_H), right])

    width = max(sum(p.shape[1] for p in row) for row in panels)
    rows_img = []
    for row in panels:
        row_w = sum(p.shape[1] for p in row)
        canvas = np.full((row[0].shape[0], width, 3), 255, np.uint8)
        x = (width - row_w) // 2
        for p in row:
            canvas[:, x:x + p.shape[1]] = p
            x += p.shape[1] + 0
        rows_img.append(canvas)

    header = np.full((PAD * 2 + 34, width, 3), 255, np.uint8)
    cv2.putText(header, "VectoVecto document restore - before / after "
                        "(real pipeline output)", (PAD, PAD + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    foot = np.full((76, width, 3), 255, np.uint8)
    cv2.putText(foot, "Green = accepted  |  amber = uncertain  |  red = "
                      "digit conflict with the alternative reading.",
                (PAD, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1,
                cv2.LINE_AA)
    cv2.putText(foot, "Letterpress page: heiDATA doi:10.11588/data/EGOKEI "
                      "(CC BY 4.0).  Photo row: this repository's synthetic "
                      "fixture under a simulated phone shadow.",
                (PAD, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1,
                cv2.LINE_AA)

    parts = [header]
    for i, row_img in enumerate(rows_img):
        parts.append(row_img)
        parts.append(np.full((GAP, width, 3), 255, np.uint8))
    parts.append(foot)
    fig = np.vstack(parts)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc_data.imwrite_safe(OUT, fig)
    print(f"wrote {OUT} ({fig.shape[1]}x{fig.shape[0]})")
    return OUT


if __name__ == "__main__":
    build()
