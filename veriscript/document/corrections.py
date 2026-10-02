"""
corrections.py — the verify -> fix -> re-export loop (docs/CORRECTIONS.md).

A corrections file is a small JSON document mapping page stems to per-token
fixes (`index` + `bbox` + `original` + `corrected` + `action`). Applying one:

* rewrites the token text in place (or confirms the engine reading) and clears
  the reviewed token from the queue;
* records provenance (`orig_text`, `corrected_by="human"`, `text_source`);
* never silently drops a correction — unmatched entries are counted and
  reported;
* lets the normal exporters write corrected artifacts (`corrected.*`) next to
  the untouched originals.

The optional training export packs ONLY user-marked token crops + labels into
a zip; nothing leaves the machine on its own (docs/CORRECTIONS.md §7).
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from veriscript.document.ocr import OCRResult, Token, _crop_with_pad

CORRECTIONS_KIND = "veriscript.corrections"
CORRECTIONS_VERSION = 1
MAX_CORRECTIONS = 500
MAX_TEXT_CHARS = 200
BBOX_CLOSE_RATIO = 0.75   # centers closer than this * token height
CROP_PAD_RATIO = 0.05


@dataclass
class Correction:
    index: int
    corrected: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    original: str = ""
    action: str = "changed"  # "changed" | "confirmed"


@dataclass
class CorrectionsDocument:
    pages: Dict[str, List[Correction]] = field(default_factory=dict)
    kind: str = CORRECTIONS_KIND
    version: int = CORRECTIONS_VERSION
    created: str = ""
    run_id: Optional[str] = None

    def for_page(self, stem: str) -> List[Correction]:
        """Page entries, with `*` as the single-page fallback."""
        if stem in self.pages:
            return self.pages[stem]
        return self.pages.get("*", [])


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _parse_bbox(raw) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _parse_correction(raw: Dict, where: str) -> Correction:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: every correction must be an object")
    try:
        index = int(raw.get("index", -1))
    except (TypeError, ValueError):
        index = -1
    if index < 0:
        raise ValueError(f"{where}: correction needs a non-negative `index`")
    corrected = str(raw.get("corrected", ""))
    if len(corrected) > MAX_TEXT_CHARS:
        raise ValueError(f"{where}[{index}]: corrected text exceeds "
                         f"{MAX_TEXT_CHARS} chars")
    action = str(raw.get("action", "changed"))
    if action not in ("changed", "confirmed"):
        action = "changed"
    return Correction(index=index, corrected=corrected,
                      bbox=_parse_bbox(raw.get("bbox")),
                      original=str(raw.get("original", "")), action=action)


def load_corrections(source: Union[str, dict]) -> CorrectionsDocument:
    """Load a corrections file (path or dict); a bare list means `*`."""
    if isinstance(source, dict):
        data = source
    else:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, list):
        data = {"pages": {"*": data}}
    if not isinstance(data, dict):
        raise ValueError("corrections: expected an object or a list")
    kind = data.get("kind")
    if kind not in (None, CORRECTIONS_KIND):
        raise ValueError(f"corrections: unknown kind {kind!r}")
    pages_raw = data.get("pages") or {}
    if not isinstance(pages_raw, dict):
        raise ValueError("corrections: `pages` must be an object")
    pages: Dict[str, List[Correction]] = {}
    total = 0
    for stem, items in pages_raw.items():
        if not isinstance(items, list):
            raise ValueError(f"corrections: page {stem!r} must be a list")
        parsed = [_parse_correction(x, f"page {stem!r}")
                  for x in items]
        total += len(parsed)
        pages[str(stem)] = parsed
    if total > MAX_CORRECTIONS:
        raise ValueError(f"corrections: more than {MAX_CORRECTIONS} entries")
    return CorrectionsDocument(
        pages=pages,
        version=int(data.get("version", CORRECTIONS_VERSION) or 0),
        created=str(data.get("created", "")),
        run_id=data.get("run_id"))


def dump_corrections(doc: CorrectionsDocument, path: str) -> str:
    payload = {
        "kind": doc.kind,
        "version": doc.version,
        "created": doc.created or _utcnow(),
        "run_id": doc.run_id,
        "pages": {
            stem: [
                {"index": c.index, "bbox": list(c.bbox) if c.bbox else None,
                 "original": c.original, "corrected": c.corrected,
                 "action": c.action}
                for c in items
            ]
            for stem, items in doc.pages.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox_close(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    (ax, ay), (bx, by) = _bbox_center(a), _bbox_center(b)
    tol = max(8.0, BBOX_CLOSE_RATIO * max(1, b[3] - b[1]))
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= tol


def _center_inside(inner: Tuple[int, int, int, int],
                   outer: Tuple[int, int, int, int]) -> bool:
    cx, cy = _bbox_center(inner)
    x0, y0, x1, y1 = outer
    pad = max(8.0, 0.5 * (outer[3] - outer[1]))
    return (x0 - pad) <= cx <= (x1 + pad) and (y0 - pad) <= cy <= (y1 + pad)


def apply_corrections(result: OCRResult,
                      corrections: Sequence[Correction]) -> Dict:
    """Apply corrections to `result.tokens` in place; rebuild `result.text`.

    Matching (docs/CORRECTIONS.md §3.2): index first with an `original`/bbox
    cross-check, then bbox-center + text fallback. Unmatched entries are
    counted, never silently dropped. Returns stats.
    """
    stats = {"changed": 0, "confirmed": 0, "skipped": 0,
             "skipped_indices": [], "reviewed": 0}
    tokens = result.tokens
    for corr in corrections:
        tok: Optional[Token] = None
        if 0 <= corr.index < len(tokens):
            cand = tokens[corr.index]
            if (not corr.original or corr.original == cand.text
                    or (corr.bbox is not None
                        and _bbox_close(corr.bbox, cand.bbox))):
                tok = cand
        if tok is None and corr.bbox is not None:
            for cand in tokens:
                if corr.original and cand.text != corr.original:
                    continue
                if _center_inside(cand.bbox, corr.bbox):
                    tok = cand
                    break
        if tok is None:
            stats["skipped"] += 1
            stats["skipped_indices"].append(corr.index)
            continue
        if corr.action == "confirmed" or corr.corrected == tok.text:
            tok.flags = []
            tok.corrected_by = "human"
            stats["confirmed"] += 1
        else:
            if tok.orig_text is None:
                tok.orig_text = tok.text
            tok.text = corr.corrected
            tok.text_source = "human"
            tok.corrected_by = "human"
            tok.flags = []
            stats["changed"] += 1
    result.text = "\n".join(t.text for t in tokens)
    stats["reviewed"] = stats["changed"] + stats["confirmed"]
    return stats


def result_from_ocr_json(path: str) -> OCRResult:
    """Rebuild token objects from an exported `ocr.json`.

    The durable input shared by the CLI, scripts and the web API: corrections
    are always applied to the original reading, so a batch can be re-sent
    idempotently (docs/CORRECTIONS.md §5).
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    tokens: List[Token] = []
    for item in payload.get("tokens", []):
        bbox = item.get("bbox") or [0, 0, 1, 1]
        tokens.append(Token(
            text=str(item.get("text", "")),
            conf=float(item.get("conf") or 0.0),
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            granularity=item.get("granularity", "line"),
            backend=payload.get("backend", "?"),
            flags=list(item.get("flags") or []),
            alt_text=item.get("alt_text"),
            repass_text=item.get("repass_text"),
            repass_conf=item.get("repass_conf"),
            text_source=item.get("text_source", "backend"),
            orig_text=item.get("original_text"),
            corrected_by=item.get("corrected_by")))
    return OCRResult(text="\n".join(t.text for t in tokens), tokens=tokens,
                     backend=payload.get("backend", "?"), meta={})


def apply_and_export(out_dir: str, stem: str, image_bgr: np.ndarray,
                     result: OCRResult, corrections: Sequence[Correction],
                     dpi: Optional[int] = None, *,
                     make_pdf: bool = True, make_overlay: bool = False,
                     make_txt: bool = True, make_json: bool = True,
                     make_md: bool = True,
                     overlay_source: Optional[np.ndarray] = None) -> Dict:
    """Apply corrections and write the corrected artifacts (never overwrites
    the originals). Returns `{"stats": …, "files": …}`; no files are written
    when nothing was reviewed."""
    stats = apply_corrections(result, corrections)
    files: Dict[str, str] = {}
    if stats["reviewed"]:
        from veriscript.document.export import export_document_outputs
        files = export_document_outputs(
            out_dir, stem, image_bgr, result, dpi=dpi,
            make_pdf=make_pdf, make_overlay=make_overlay,
            make_txt=make_txt, make_json=make_json, make_md=make_md,
            overlay_source=overlay_source)
    return {"stats": stats, "files": files}


def corrections_record(page: str, corrections: Sequence[Correction],
                       stats: Dict, run_id: Optional[str] = None) -> Dict:
    """The applied-corrections record written next to the artifacts."""
    return {
        "kind": CORRECTIONS_KIND,
        "version": CORRECTIONS_VERSION,
        "created": _utcnow(),
        "run_id": run_id,
        "page": page,
        "stats": stats,
        "applied": [
            {"index": c.index, "bbox": list(c.bbox) if c.bbox else None,
             "original": c.original, "corrected": c.corrected,
             "action": c.action}
            for c in corrections
        ],
    }


def write_corrections_record(path: str, record: Dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# privacy-first training export
# ---------------------------------------------------------------------------

_EXPORT_README = """\
VeriScript correction export
============================

This archive was created by its owner from one OCR run. It contains ONLY the
token crops they explicitly marked shareable, plus their corrected text.

  corrections.json  the corrections (page, bbox, original, corrected)
  labels.tsv        crop file -> verified text (recognizer training format)
  crops/            one PNG per marked token, cut from the run image

No full-page images are included. Nothing was uploaded anywhere: this file is
the only artifact that left the machine and the owner created it.

To use it for Devanagari digit/line training, convert crops/ + labels.tsv
into the `lines/ + labels.tsv` layout of digit_lines_v1 or pass it through
scripts/export_training_data.py --real-dir.
"""


def build_corrections_zip(out_path: str, image_bgr: np.ndarray, page: str,
                          corrections: Sequence[Correction],
                          share_indices: Sequence[int]) -> str:
    """Pack labelled crops for `share_indices` ONLY (docs/CORRECTIONS.md §7)."""
    by_index = {c.index: c for c in corrections}
    rows: List[Tuple[str, str]] = []
    crops: List[Tuple[str, np.ndarray]] = []
    for idx in sorted(set(int(i) for i in share_indices)):
        corr = by_index.get(idx)
        if corr is None or corr.bbox is None:
            continue
        crop = _crop_with_pad(image_bgr, corr.bbox, pad_ratio=CROP_PAD_RATIO)
        if crop.size == 0:
            continue
        label = (corr.corrected if corr.action != "confirmed"
                 and corr.corrected else corr.original or corr.corrected)
        name = f"{page}_{idx:04d}.png"
        crops.append((name, crop))
        rows.append((name, label))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", _EXPORT_README)
        z.writestr("corrections.json", json.dumps({
            "kind": CORRECTIONS_KIND,
            "version": CORRECTIONS_VERSION,
            "created": _utcnow(),
            "page": page,
            "shared": [r[0] for r in rows],
            "items": [
                {"index": c.index, "bbox": list(c.bbox) if c.bbox else None,
                 "original": c.original, "corrected": c.corrected,
                 "action": c.action, "shared": c.index in set(share_indices)}
                for c in corrections
            ],
        }, indent=2, ensure_ascii=False))
        z.writestr("labels.tsv",
                   "".join(f"{name}\t{text}\n" for name, text in rows))
        for name, crop in crops:
            ok, buf = cv2.imencode(".png", crop)
            if ok:
                z.writestr(f"crops/{name}", buf.tobytes())
    return out_path
