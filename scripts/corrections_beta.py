#!/usr/bin/env python3
"""corrections_beta.py — the correction-loop beta harness (docs/CORRECTIONS.md §8).

    prepare : render one NON-FROZEN source page, run the shipped pipeline, and
              save a reviewable session (page images, queue, sheets, provenance).
    apply   : apply a reviewed corrections file; write corrected artifacts, the
              training zip and session stats.
    report  : aggregate the sessions into beta_report.json (≥20% acted-on bar).
    import  : convert reviewed digit-bearing corrections into a staged
              target-domain dataset (lines/ + labels.tsv + manifest) for human
              audit before any training use.

The manifest guard refuses frozen page ids, so the beta can never touch an
evaluation page.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.core import data as doc_data  # noqa: E402
from veriscript.document.corrections import (apply_and_export,  # noqa: E402
                                             build_corrections_zip,
                                             corrections_record,
                                             load_corrections,
                                             result_from_ocr_json,
                                             write_corrections_record)
from veriscript.document.ocr import (_crop_with_pad,  # noqa: E402
                                     review_queue, token_risk)

MONTAGE_CELLS = 24
MONTAGE_COLS = 2


def _frozen_ids() -> set:
    ids = set()
    for path in glob.glob(os.path.join(BASE_DIR, "evals", "manifests", "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for entry in data.get("entries", []):
            if "id" in entry:
                ids.add(entry["id"])
    return ids


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: str, payload) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _render_page(source: str, page_index: int, dpi: int) -> np.ndarray:
    for idx, img, _gt in doc_data.pdf_to_pages(source, dpi=dpi):
        if idx == page_index:
            return img
    raise SystemExit(f"page {page_index} not found in {source}")


def _build_sheets(session_dir: str, display: np.ndarray,
                  queue: List[Dict]) -> List[str]:
    """Review aids: one contact sheet per <=24 tokens, index-labelled."""
    cells: List[Tuple[int, np.ndarray]] = []
    for item in queue:
        crop = _crop_with_pad(display, tuple(item["bbox"]), pad_ratio=0.12)
        if crop.size:
            cells.append((int(item["index"]), crop))
    written = []
    for start in range(0, len(cells), MONTAGE_CELLS):
        chunk = cells[start:start + MONTAGE_CELLS]
        cols = MONTAGE_COLS if len(chunk) > 1 else 1
        rows = (len(chunk) + cols - 1) // cols
        cw = max(c.shape[1] for _, c in chunk) + 90
        ch = max(c.shape[0] for _, c in chunk) + 10
        canvas = np.full((rows * ch + 10, cols * cw + 10, 3), 255, np.uint8)
        for k, (idx, crop) in enumerate(chunk):
            r, c = divmod(k, cols)
            x = 10 + c * cw + 80
            y = 10 + r * ch
            canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            cv2.putText(canvas, str(idx), (10 + c * cw, y + crop.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (150, 0, 0), 3,
                        cv2.LINE_AA)
        path = os.path.join(session_dir, f"montage_{start // MONTAGE_CELLS:02d}.png")
        cv2.imwrite(path, canvas)
        written.append(path)
    return written


def prepare(source: str, page_index: int, out_dir: str, dpi: int = 300,
            lang: str = "ne") -> Dict:
    stem = os.path.splitext(os.path.basename(source))[0]
    page_id = f"{stem}_p{page_index:03d}"
    if page_id in _frozen_ids():
        raise SystemExit(f"{page_id} is a frozen evaluation page - refused")
    if os.path.exists(os.path.join(out_dir, "session.json")):
        raise SystemExit(f"{out_dir} already prepared")

    from veriscript.document.pipeline import run_document_pipeline

    os.makedirs(out_dir, exist_ok=True)
    img = _render_page(source, page_index, dpi)
    t0 = time.time()
    result = run_document_pipeline(
        img, backend="rapidocr", lang=lang, out_dir=out_dir, stem="page",
        dpi=dpi, make_pdf=True, make_overlay=True, make_txt=True,
        make_json=True, make_md=True, deva_lines="off")
    elapsed = round(time.time() - t0, 2)
    cv2.imwrite(os.path.join(out_dir, "page_input.png"), img)
    cv2.imwrite(os.path.join(out_dir, "page_display.png"), result.display_bgr)

    index_of = {id(t): i for i, t in enumerate(result.ocr.tokens)}
    queue = [{
        "index": index_of[id(t)],
        "text": t.text,
        "alt_text": t.alt_text,
        "flags": list(t.flags),
        "conf": round(float(t.conf), 2),
        "risk": round(token_risk(t), 2),
        "bbox": list(t.bbox),
    } for t in review_queue(result.ocr.tokens)]
    _write_json(os.path.join(out_dir, "queue.json"), queue)
    sheets = _build_sheets(out_dir, result.display_bgr, queue)
    with open(os.path.join(out_dir, "sheet.tsv"), "w", encoding="utf-8",
              newline="\n") as f:
        for item in queue:
            flags = ",".join(item["flags"])
            text = item["text"].replace("\t", " ").replace("\n", " ")
            alt = (item["alt_text"] or "").replace("\t", " ").replace("\n", " ")
            f.write(f"{item['index']}\t{text}\t{alt}\t{flags}\t"
                    f"{item['conf']}\t{item['risk']}\n")
    _write_json(os.path.join(out_dir, "corrections_template.json"), {
        "kind": "veriscript.corrections", "version": 1,
        "run_id": page_id, "pages": {"*": []}})
    session = {
        "page_id": page_id, "source": source, "page_index": page_index,
        "dpi": dpi, "lang": lang, "engine": "rapidocr", "deva_lines": "off",
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_seconds": elapsed,
        "tokens": len(result.ocr.tokens), "queued": len(queue),
        "montages": [os.path.basename(p) for p in sheets],
    }
    _write_json(os.path.join(out_dir, "session.json"), session)
    print(f"[beta] prepared {page_id}: {session['tokens']} tokens, "
          f"{session['queued']} queued, {elapsed:.1f}s, "
          f"{len(sheets)} sheet(s) -> {out_dir}")
    return session


def compose(session_dir: str, edits_path: str,
            out_path: Optional[str] = None) -> Dict:
    """Build a corrections file from an `edits.tsv` (index, action, text).

    The queue is the source of `bbox`/`original`, so the file can never carry a
    stale bbox or a mistyped original.
    """
    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as f:
        session = json.load(f)
    with open(os.path.join(session_dir, "queue.json"), encoding="utf-8") as f:
        queue = {int(q["index"]): q for q in json.load(f)}
    items = []
    with open(edits_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise SystemExit(f"{edits_path}:{line_no}: expected "
                                 f"index<TAB>action[<TAB>text]")
            try:
                idx = int(parts[0])
            except ValueError:
                raise SystemExit(f"{edits_path}:{line_no}: bad index")
            action = parts[1].strip() or "changed"
            if action not in ("changed", "confirmed"):
                raise SystemExit(f"{edits_path}:{line_no}: bad action")
            text = parts[2] if len(parts) > 2 else ""
            q = queue.get(idx)
            if q is None:
                raise SystemExit(f"{edits_path}:{line_no}: index {idx} is not "
                                 f"in this session's queue")
            items.append({
                "index": idx, "bbox": q["bbox"], "original": q["text"],
                "corrected": q["text"] if action == "confirmed" else text,
                "action": action,
            })
    out = out_path or os.path.join(session_dir, "review", "corrections.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    _write_json(out, {"kind": "veriscript.corrections", "version": 1,
                      "run_id": session["page_id"], "pages": {"*": items}})
    print(f"[beta] composed {len(items)} corrections -> {out}")
    return {"items": items, "path": out}


def apply_review(session_dir: str, corrections_path: str) -> Dict:
    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as f:
        session = json.load(f)
    with open(os.path.join(session_dir, "queue.json"), encoding="utf-8") as f:
        queued = json.load(f)
    corr = load_corrections(corrections_path).for_page("*")
    display = cv2.imread(os.path.join(session_dir, "page_display.png"))
    if display is None:
        raise SystemExit(f"missing page_display.png in {session_dir}")
    result = result_from_ocr_json(os.path.join(session_dir, "page.json"))
    applied = apply_and_export(
        session_dir, "corrected", display, result, corr,
        dpi=int(session.get("dpi", 300)), make_overlay=False,
        overlay_source=display)
    stats = applied["stats"]
    if stats["reviewed"]:
        write_corrections_record(
            os.path.join(session_dir, "beta_corrections.json"),
            corrections_record(session["page_id"], corr, stats))
    build_corrections_zip(os.path.join(session_dir, "corrections.zip"),
                          display, session["page_id"], corr,
                          share_indices=[c.index for c in corr])
    queue_after = len(review_queue(result.tokens))
    session_stats = {
        "page_id": session["page_id"], **stats,
        "queued": len(queued), "queue_after": queue_after,
        "acted_fraction": (round((stats["changed"] + stats["confirmed"])
                                 / len(queued), 4) if queued else None),
        "bar_20pct": (bool(queued and (stats["changed"] + stats["confirmed"])
                           >= 0.2 * len(queued))),
    }
    _write_json(os.path.join(session_dir, "session_stats.json"), session_stats)
    print(f"[beta] applied {session['page_id']}: {stats['changed']} changed / "
          f"{stats['confirmed']} confirmed / {stats['skipped']} skipped "
          f"({session_stats['acted_fraction']} of {len(queued)} queued, "
          f"queue now {queue_after})")
    return session_stats


def report(beta_dir: str) -> Dict:
    rows = []
    for path in sorted(glob.glob(os.path.join(beta_dir, "*",
                                              "session_stats.json"))):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    sessions = len(rows)
    queued = sum(r["queued"] for r in rows)
    acted = sum(r["changed"] + r["confirmed"] for r in rows)
    payload = {
        "sessions": sessions,
        "queued": queued,
        "changed": sum(r["changed"] for r in rows),
        "confirmed": sum(r["confirmed"] for r in rows),
        "skipped": sum(r["skipped"] for r in rows),
        "acted": acted,
        "acted_fraction": round(acted / queued, 4) if queued else None,
        "sessions_meeting_20pct": sum(1 for r in rows if r["bar_20pct"]),
        "queue_after": sum(r["queue_after"] for r in rows),
        "rows": rows,
    }
    _write_json(os.path.join(beta_dir, "beta_report.json"), payload)
    print(f"\n=== BETA ({sessions} sessions) ===")
    print(f"queued {queued} · acted {acted} "
          f"({payload['acted_fraction']:.1%}) · changed {payload['changed']} "
          f"· confirmed {payload['confirmed']} · skipped {payload['skipped']}")
    print(f"sessions meeting the 20% bar: "
          f"{payload['sessions_meeting_20pct']}/{sessions}")
    return payload


def import_dataset(beta_dir: str, dest: str, force: bool = False) -> Dict:
    labels_path = os.path.join(dest, "labels.tsv")
    if os.path.exists(labels_path) and not force:
        raise SystemExit(f"{labels_path} exists; pass --force to overwrite")
    lines_dir = os.path.join(dest, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    entries = []
    labels = []
    for session_dir in sorted(glob.glob(os.path.join(beta_dir, "*"))):
        corrected = os.path.join(session_dir, "corrected.json")
        display_path = os.path.join(session_dir, "page_display.png")
        if not (os.path.isfile(corrected) and os.path.isfile(display_path)):
            continue
        with open(corrected, encoding="utf-8") as f:
            payload = json.load(f)
        display = cv2.imread(display_path)
        page_id = os.path.basename(session_dir)
        for item in payload.get("tokens", []):
            if item.get("corrected_by") != "human":
                continue
            text = str(item.get("text", ""))
            if not text or not any(ch.isdigit() for ch in text):
                continue
            bbox = item.get("bbox")
            crop = _crop_with_pad(display, tuple(bbox), pad_ratio=0.08)
            if crop.size == 0:
                continue
            name = f"{page_id}_{int(item.get('index', len(labels))):04d}.png"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            with open(os.path.join(lines_dir, name), "wb") as f:
                f.write(buf.tobytes())
            labels.append((name, text))
            entries.append({
                "file": name, "text": text,
                "original": item.get("original_text"),
                "page": page_id, "bbox": bbox,
                "flags": item.get("flags", []),
                "conf": item.get("conf"),
            })
    with open(labels_path, "w", encoding="utf-8", newline="\n") as f:
        for name, text in labels:
            f.write(f"{name}\t{text}\n")
    manifest = {
        "name": os.path.basename(dest.rstrip("/\\")),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "corrections beta (docs/CORRECTIONS.md)",
        "reviewed_by": "agent beta review - NOT human GT; audit before "
                       "training use",
        "n_labels": len(labels),
        "labels_sha256": _sha256(labels_path),
        "entries": entries,
    }
    _write_json(os.path.join(dest, "import_manifest.json"), manifest)
    print(f"[beta] imported {len(labels)} digit-bearing corrections -> {dest} "
          f"(labels sha256 {manifest['labels_sha256'][:12]})")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="correction-loop beta harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--page", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--lang", default="ne")

    a = sub.add_parser("apply")
    a.add_argument("--session", required=True)
    a.add_argument("--corrections", required=True)

    c = sub.add_parser("compose")
    c.add_argument("--session", required=True)
    c.add_argument("--edits", required=True)
    c.add_argument("--out", default=None)

    r = sub.add_parser("report")
    r.add_argument("--beta-dir", required=True)

    i = sub.add_parser("import")
    i.add_argument("--beta-dir", required=True)
    i.add_argument("--dest", required=True)
    i.add_argument("--force", action="store_true")

    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.source, args.page, args.out, dpi=args.dpi, lang=args.lang)
    elif args.cmd == "apply":
        apply_review(args.session, args.corrections)
    elif args.cmd == "compose":
        compose(args.session, args.edits, args.out)
    elif args.cmd == "report":
        report(args.beta_dir)
    elif args.cmd == "import":
        import_dataset(args.beta_dir, args.dest, force=args.force)


if __name__ == "__main__":
    main()
