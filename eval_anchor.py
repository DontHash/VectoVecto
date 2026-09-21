"""
eval_anchor.py — modern-Nepali anchor: consensus + human-verified pages (N5).

The v2 text layers are systematically corrupt (page mean 22.6% invalid
tokens, wrong ToUnicode maps), so no absolute CER can be quoted from them.
This harness builds the anchor:

  * build : for every v2 page, produce three independent readings
            (PDF text layer / RapidOCR / bodhan line re-reads), score bag
            agreement, classify pages as consensus-anchored (>= agreement
            bar) or worksheet pages (budgeted, stratified across documents).
            Writes out/anchor_readings.json, out/anchor/worksheet.html and
            pre-fills out/anchor/verified/<page>.txt with the RapidOCR reading
            so the human 15-minute pass only edits errors.
  * score : score the human-verified pages -> unbiased CER/bagCER + bootstrap
            CI; report the consensus set separately (agreement-bounded, with
            the selection-bias caveat).

Usage:
    python eval_anchor.py --data-dir data/doc_eval/nepali_pdf_v2 --frozen \
        evals/manifests/nepali_pdf_v2.json --build --out out/anchor
    python eval_anchor.py --score --verified out/anchor/verified \
        --readings out/anchor/anchor_readings.json --json out/anchor_score.json
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from document_ocr import ocr_page  # noqa: E402

AGREEMENT_BAR = 0.95
DEFAULT_BUDGET = 8


def select_worksheet(rows: List[Dict], threshold: float = AGREEMENT_BAR,
                     budget: int = DEFAULT_BUDGET):
    """Split pages into (consensus, worksheet); worksheet is doc-stratified.

    Deterministic: within each document, lowest-agreement pages first, then
    round-robin across documents until the budget is filled.
    """
    consensus = [r["page"] for r in rows if r["agreement_rate"] >= threshold]
    rest = [r for r in rows if r["agreement_rate"] < threshold]
    by_doc: Dict[str, List[Dict]] = {}
    for r in sorted(rest, key=lambda r: (r["agreement_rate"], r["page"])):
        by_doc.setdefault(r["doc"], []).append(r)
    picked: List[str] = []
    while len(picked) < budget and any(by_doc.values()):
        for doc in sorted(by_doc):
            if by_doc[doc] and len(picked) < budget:
                picked.append(by_doc[doc].pop(0)["page"])
    return consensus, picked


def write_verified_stubs(rows: List[Dict], out_dir: str,
                         picked: List[str]) -> List[str]:
    """Pre-fill one editable GT file per worksheet page (never clobbers)."""
    os.makedirs(out_dir, exist_ok=True)
    by_page = {r["page"]: r for r in rows}
    written = []
    for page in picked:
        path = os.path.join(out_dir, f"{page}.txt")
        if os.path.exists(path):
            continue
        text = by_page[page]["readings"]["rapidocr"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        written.append(path)
    return written


def _img_data_uri(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def mark_disagreements(text: str, other: str) -> str:
    """HTML-escape `text`, wrapping tokens absent from `other` in <mark>."""
    other_set = {doc_metrics._norm_tok(t) for t in doc_metrics.tokens_of(other)
                 if doc_metrics._norm_tok(t)}
    out = []
    for tok in doc_metrics.tokens_of(text):
        esc = html_mod.escape(tok)
        if doc_metrics._norm_tok(tok) and doc_metrics._norm_tok(tok) not in other_set:
            out.append(f"<mark>{esc}</mark>")
        else:
            out.append(esc)
    return " ".join(out)


def render_html(rows: List[Dict], out_path: str) -> str:
    """Self-contained worksheet: page image + engine readings, disagreements marked.

    The corrupt text layer is kept in a collapsed block for the record; the
    human pass compares RapidOCR (the system under test, pre-filled in
    `verified/<page>.txt`) against bodhan and edits only the marked spots.
    """
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Anchor worksheet</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}"
        ".page{border:1px solid #ccc;border-radius:8px;margin:18px 0;padding:12px;}"
        "img{max-width:100%;max-height:520px;border:1px solid #eee;}"
        "table{border-collapse:collapse;width:100%;}"
        "td,th{vertical-align:top;text-align:left;padding:6px;"
        "border-top:1px solid #eee;font-size:14px;}"
        "mark{background:#ffe08a;}"
        ".rate{color:#555;font-size:13px;}"
        "details{margin-top:6px;color:#777;font-size:13px;}</style>",
        "<h1>Anchor worksheet</h1>",
        "<p>For each page: read the image, edit "
        "<code>verified/&lt;page&gt;.txt</code> (pre-filled with RapidOCR). "
        "Highlighted tokens are where the two engines disagree - those are the "
        "only spots that need a decision; agreement spots are usually right.</p>",
    ]
    for r in rows:
        uri = _img_data_uri(r.get("image", ""))
        img_tag = f"<img src='{uri}'>" if uri else "<em>no image</em>"
        rd = r.get("readings", {})
        dis = ", ".join(r.get("agreement", {}).get("disagreements", [])[:12])
        parts.append(
            f"<div class='page' id='{html_mod.escape(r['page'])}'>"
            f"<h2>{html_mod.escape(r['page'])}</h2>"
            f"<div class='rate'>engine agreement {r['agreement_rate']:.1%} | "
            f"disagreements: {html_mod.escape(dis) or 'none'}</div>"
            f"{img_tag}"
            "<table><tr><th>RapidOCR (edit this in verified/*.txt)</th>"
            "<th>bodhan</th></tr>"
            f"<tr><td>{mark_disagreements(rd.get('rapidocr', ''), rd.get('bodhan', ''))}</td>"
            f"<td>{mark_disagreements(rd.get('bodhan', ''), rd.get('rapidocr', ''))}</td></tr></table>"
            f"<details><summary>corrupt text layer (record only)</summary>"
            f"{html_mod.escape(rd.get('text_layer', ''))}</details>"
            "</div>")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return out_path


def _finalize(rows: List[Dict], out_dir: str, threshold: float,
              budget: int) -> Dict:
    """Recompute agreements from stored readings, write all anchor artifacts."""
    for r in rows:
        rd = r["readings"]
        r["agreement"] = doc_metrics.three_way_agreement(
            [rd["rapidocr"], rd["bodhan"]])
        r["agreement_rate"] = r["agreement"]["agreement_rate"]
        r["text_layer_agreement"] = doc_metrics.three_way_agreement(
            [rd["text_layer"], rd["rapidocr"]])["agreement_rate"]
    consensus, picked = select_worksheet(rows, threshold, budget)
    os.makedirs(out_dir, exist_ok=True)
    readings_path = os.path.join(out_dir, "anchor_readings.json")
    with open(readings_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "budget": budget,
                   "consensus": consensus, "worksheet": picked,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    worksheet_rows = [r for r in rows if r["page"] in set(picked)]
    render_html(worksheet_rows, os.path.join(out_dir, "worksheet.html"))
    render_contact_sheet(worksheet_rows, picked,
                         os.path.join(out_dir, "worksheet.png"))
    written = write_verified_stubs(rows, os.path.join(out_dir, "verified"),
                                   picked)
    mean_consensus = (float(np.mean([r["agreement_rate"] for r in rows
                                     if r["page"] in set(consensus)]))
                      if consensus else None)
    line = (f"[anchor] pages {len(rows)} | consensus {len(consensus)}"
            + (f" (mean agreement {mean_consensus:.1%})" if mean_consensus is not None else "")
            + f" | worksheet {len(picked)}")
    print("\n" + line)
    print(f"[anchor] readings: {readings_path}")
    print(f"[anchor] worksheet: {os.path.join(out_dir, 'worksheet.html')}")
    print(f"[anchor] pre-filled GT stubs: {len(written)} "
          f"-> {os.path.join(out_dir, 'verified')}")
    return {"rows": rows, "consensus": consensus, "worksheet": picked}


def render_contact_sheet(rows: List[Dict], picked: List[str], out_path: str,
                         thumb_w: int = 420) -> str:
    """Overview PNG: worksheet page thumbnails with id + agreement."""
    import cv2

    tiles = []
    for page in picked:
        r = next((x for x in rows if x["page"] == page), None)
        if r is None or not os.path.exists(r.get("image", "")):
            continue
        img = doc_data.imread_safe(r["image"])
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        tile = cv2.resize(img, (thumb_w, max(1, int(img.shape[0] * scale))))
        banner = np.full((26, thumb_w, 3), 255, dtype=np.uint8)
        cv2.putText(banner, f"{page[:44]}  {r['agreement_rate']:.0%}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        tiles.append(np.vstack([banner, tile]))
    if not tiles:
        return out_path
    h = max(t.shape[0] for t in tiles)
    padded = [np.vstack([t, np.full((h - t.shape[0], thumb_w, 3), 255,
                                   dtype=np.uint8)]) for t in tiles]
    per_row = 3
    rows_img = []
    for i in range(0, len(padded), per_row):
        chunk = padded[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.full_like(padded[0], 255))
        rows_img.append(np.hstack(chunk))
    sheet = np.vstack(rows_img)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc_data.imwrite_safe(out_path, sheet)
    return out_path


def _crop(img: np.ndarray, bbox, pad: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return img[max(0, y0 - dy):min(h, y1 + dy), max(0, x0 - dx):min(w, x1 + dx)]


def build(data_dir: str, out_dir: str, frozen: Optional[str] = None,
          lang: str = "ne", threshold: float = AGREEMENT_BAR,
          budget: int = DEFAULT_BUDGET, use_bodhan: bool = True,
          limit: int = 0) -> Dict:
    if frozen:
        from eval_freeze import verify_manifest
        ok, problems = verify_manifest(frozen, data_dir=data_dir)
        if not ok:
            for p in problems[:10]:
                print("  [frozen] DRIFT:", p)
            raise SystemExit("frozen drift - refusing to build the anchor")
        print(f"[anchor] frozen verified: {os.path.basename(frozen)}")

    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"]
    if limit:
        entries = entries[:limit]

    verifier = None
    if use_bodhan:
        from document_verifier import get_digit_verifier
        verifier = get_digit_verifier("bodhan")
        print("[anchor] bodhan verifier loaded (line re-reads)")

    rows: List[Dict] = []
    try:
        for i, e in enumerate(entries, 1):
            img = doc_data.imread_safe(e["_degraded_path"])
            if img is None:
                continue
            gt = open(e["_gt_path"], encoding="utf-8").read()
            rec = ocr_page(img, backend="rapidocr", lang=lang)
            bodhan_text = ""
            if verifier is not None and rec.tokens:
                crops = [_crop(img, t.bbox) for t in rec.tokens]
                try:
                    reads = [x for x in verifier(crops) if x]
                    bodhan_text = " ".join(str(x) for x in reads)
                except Exception as ex:  # noqa: BLE001
                    print(f"  [anchor] bodhan failed on {e['id']}: {ex}")
            agreement = doc_metrics.three_way_agreement([rec.text, bodhan_text])
            rows.append({
                "page": e["id"], "doc": e["id"].rsplit("_p", 1)[0],
                "image": e["_degraded_path"],
                "agreement_rate": agreement["agreement_rate"],
                "agreement": agreement,
                "gt_invalid_rate": e.get("gt_invalid_token_rate"),
                "readings": {"text_layer": gt, "rapidocr": rec.text,
                             "bodhan": bodhan_text},
            })
            print(f"  [{i}/{len(entries)}] {e['id'][:40]} "
                  f"agreement {agreement['agreement_rate']:.1%}", flush=True)
    finally:
        if verifier is not None:
            verifier.close()

    return _finalize(rows, out_dir, threshold, budget)


def score(verified_dir: str, readings_path: str,
          json_path: Optional[str] = None) -> Dict:
    """Score human-verified pages; report the consensus set separately."""
    data = json.load(open(readings_path, encoding="utf-8"))
    rows = {r["page"]: r for r in data["rows"]}
    worksheet = set(data.get("worksheet", []))
    per_page = []
    skipped = []
    for page, r in sorted(rows.items()):
        path = os.path.join(verified_dir, f"{page}.txt")
        if not os.path.exists(path):
            continue
        if page not in worksheet:
            skipped.append(page)  # stale pre-fill from an earlier selection
            continue
        gt = open(path, encoding="utf-8").read().strip()
        hyp = r["readings"]["rapidocr"]
        vgs = doc_metrics.valid_gt_stats(gt, hyp)
        per_page.append({
            "page": page, "doc": r["doc"],
            "cer": doc_metrics.cer(gt, hyp),
            "cer_bag": doc_metrics.cer_bag(gt, hyp),
            "digit_cer": doc_metrics.digit_cer(gt, hyp),
            "valid_recall": vgs["valid_recall"],
            "unmatched_rate": vgs["unmatched_rate"],
            "seconds": None,
        })
    summary: Dict = {"verified_pages": len(per_page)}
    if per_page:
        cers = [p["cer"] for p in per_page]
        bags = [p["cer_bag"] for p in per_page]
        digs = [p["digit_cer"] for p in per_page if p["digit_cer"] is not None]
        summary.update({
            "cer": float(np.mean(cers)),
            "cer_bag": float(np.mean(bags)),
            "digit_cer": float(np.mean(digs)) if digs else None,
            "cer_ci": list(doc_metrics.bootstrap_ci(cers, seed=71)),
            "cer_bag_ci": list(doc_metrics.bootstrap_ci(bags, seed=72)),
        })
        summary["consensus_pages"] = len(data["consensus"])
        summary["consensus_mean_agreement"] = (
            float(np.mean([rows[p]["agreement_rate"] for p in data["consensus"]]))
            if data["consensus"] else None)
    if skipped:
        print(f"[anchor] ignored {len(skipped)} stale pre-fill(s) outside the "
              f"worksheet: {', '.join(skipped[:4])}")
    result = {"summary": summary, "per_page": per_page,
              "skipped_stale": skipped,
              "consensus": data["consensus"], "worksheet": data["worksheet"]}
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n=== ANCHOR SCORE (human-verified pages) ===")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    for p in per_page:
        print(f"  {p['page'][:40]:<42} CER {p['cer']:.3f} "
              f"bagCER {p['cer_bag']:.3f} valid_recall {p['valid_recall']:.3f}")
    if not per_page:
        print("  no verified files found - fill "
              f"{verified_dir}/<page>.txt first")
    return result


def main():
    ap = argparse.ArgumentParser(description="Modern-Nepali anchor harness")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "out", "anchor"))
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--threshold", type=float, default=AGREEMENT_BAR)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--no-bodhan", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--reclassify", action="store_true",
                    help="recompute agreement/worksheet from stored readings "
                         "(no OCR re-run)")
    ap.add_argument("--verified", default=None)
    ap.add_argument("--readings", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.build:
        if not args.data_dir:
            raise SystemExit("--build needs --data-dir")
        build(args.data_dir, args.out, frozen=args.frozen, lang=args.lang,
              threshold=args.threshold, budget=args.budget,
              use_bodhan=not args.no_bodhan, limit=args.limit)
    if args.reclassify:
        path = args.readings or os.path.join(args.out, "anchor_readings.json")
        data = json.load(open(path, encoding="utf-8"))
        _finalize(data["rows"], args.out, args.threshold, args.budget)
    if args.score:
        verified = args.verified or os.path.join(args.out, "verified")
        readings = args.readings or os.path.join(args.out,
                                                 "anchor_readings.json")
        score(verified, readings, json_path=args.json)


if __name__ == "__main__":
    main()
