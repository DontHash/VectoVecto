"""
anchor_gemini.py — Gemini transcription for the modern-Nepali anchor (N5b).

The v2 text layers are corrupt and the local engines disagree everywhere, so
the anchor needs an independent reader. Gemini 2.5 Pro (Vertex AI, ADC
credentials) transcribes each page; the local engines corroborate; the audit
worksheet (`audit.html`) shows Gemini vs the engines with disagreements
marked, so the human pass is a review, not a transcription.

Provenance is explicit: this GT is *model-produced*, not human-produced. The
anchor numbers it enables are labeled accordingly, and the human review of the
disagreement list bounds Gemini's own error.

Usage:
    python anchor_gemini.py --data-dir data/doc_eval/nepali_pdf_v2 \
        --readings out/anchor/anchor_readings.json --out out/anchor_gemini
    python eval_anchor.py --score --verified out/anchor_gemini/verified \
        --readings out/anchor_gemini/gemini_readings.json \
        --json out/anchor_score_gemini.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402
from eval_anchor import render_html  # noqa: E402

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_LOCATION = "us-central1"

PROMPT = (
    "Transcribe every printed line of this Devanagari document image exactly "
    "as it appears, in reading order. Preserve the original spelling, "
    "punctuation and digits (do not convert Devanagari digits to ASCII). For "
    "tables, transcribe each row's cells separated by spaces, row by row. "
    "Output plain text only, one line per printed line, no commentary, no "
    "translation, no corrections."
)


def strip_images(img_bgr: np.ndarray, strips: int = 2,
                 overlap: int = 48) -> List[np.ndarray]:
    """Split a page into horizontal strips (readable resolution per call)."""
    h = img_bgr.shape[0]
    if strips <= 1:
        return [img_bgr]
    step = max(1, h // strips)
    out, y = [], 0
    while y < h:
        out.append(img_bgr[y:min(h, y + step + overlap)])
        y += step
    return out


def transcribe_page(client, img_bgr: np.ndarray, model: str = DEFAULT_MODEL,
                    strips: int = 2, prompt: str = PROMPT,
                    temperature: float = 0.0, timeout_s: float = 90.0,
                    retries: int = 2) -> str:
    """One Gemini transcription per page (strips joined with newlines).

    Each strip call gets a hard timeout and `retries` retries with backoff, so
    a hung request cannot stall the run silently.
    """
    import time

    from PIL import Image
    config = None
    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=temperature,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    except Exception:  # noqa: BLE001 - fake clients in tests
        config = None
    texts: List[str] = []
    for strip in strip_images(img_bgr, strips):
        contents = [prompt, Image.fromarray(strip[:, :, ::-1])]
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = client.models.generate_content(model=model,
                                                      contents=contents,
                                                      config=config)
                texts.append((resp.text or "").strip())
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(f"gemini call failed after {retries + 1} "
                               f"attempts: {last_err}")
    return doc_metrics.normalize_text("\n".join(t for t in texts if t))


def _client(model_location: str = DEFAULT_LOCATION, project: Optional[str] = None):
    from google import genai
    if project is None:
        try:
            import subprocess
            project = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            project = None
    return genai.Client(vertexai=True, project=project,
                        location=model_location)


def _write_payload(out_dir: str, payload: Dict) -> None:
    """Checkpoint after every page: an abort never loses finished work."""
    with open(os.path.join(out_dir, "gemini_readings.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(data_dir: str, out_dir: str,
        readings_path: Optional[str] = None, model: str = DEFAULT_MODEL,
        strips: int = 2, limit: int = 0, client=None,
        location: str = DEFAULT_LOCATION, resume: bool = False,
        timeout_s: float = 90.0, retries: int = 2) -> Dict:
    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"]
    if limit:
        entries = entries[:limit]
    engines: Dict[str, Dict] = {}
    if readings_path and os.path.exists(readings_path):
        data = json.load(open(readings_path, encoding="utf-8"))
        engines = {r["page"]: r["readings"] for r in data["rows"]}
    if client is None:
        client = _client(location)

    verified_dir = os.path.join(out_dir, "verified")
    os.makedirs(verified_dir, exist_ok=True)
    payload = {"model": model, "strips": strips, "provenance":
               "model-produced GT (Gemini via Vertex AI), audited where it "
               "disagrees with the local engines",
               "worksheet": [e["id"] for e in entries], "consensus": [],
               "rows": [], "failed": []}
    rows: List[Dict] = payload["rows"]
    for i, e in enumerate(entries, 1):
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        gt = open(e["_gt_path"], encoding="utf-8").read()
        vpath = os.path.join(verified_dir, f"{e['id']}.txt")
        if resume and os.path.exists(vpath):
            gemini_text = open(vpath, encoding="utf-8").read().strip()
            print(f"  [{i}/{len(entries)}] {e['id'][:38]:<40} (resumed)",
                  flush=True)
        else:
            try:
                gemini_text = transcribe_page(client, img, model=model,
                                              strips=strips,
                                              timeout_s=timeout_s,
                                              retries=retries)
            except Exception as ex:  # noqa: BLE001
                payload["failed"].append({"page": e["id"],
                                          "error": str(ex)[:200]})
                _write_payload(out_dir, payload)
                print(f"  [{i}/{len(entries)}] {e['id'][:38]:<40} "
                      f"FAILED: {str(ex)[:80]}", flush=True)
                continue
            with open(vpath, "w", encoding="utf-8") as f:
                f.write(gemini_text + "\n")
        eng = engines.get(e["id"], {})
        rapid = eng.get("rapidocr", "")
        bodhan = eng.get("bodhan", "")
        ag_rapid = (doc_metrics.three_way_agreement([gemini_text, rapid])
                    ["agreement_rate"] if rapid else None)
        ag_bodhan = (doc_metrics.three_way_agreement([gemini_text, bodhan])
                     ["agreement_rate"] if bodhan else None)
        vgs = doc_metrics.valid_gt_stats(gemini_text, rapid) if rapid else {}
        rows.append({
            "page": e["id"], "doc": e["id"].rsplit("_p", 1)[0],
            "image": e["_degraded_path"],
            "agreement_rate": ag_rapid if ag_rapid is not None else 0.0,
            "agreement": {"disagreements": (
                doc_metrics.three_way_agreement([gemini_text, rapid])
                ["disagreements"] if rapid else [])},
            "gemini_vs_text_layer": doc_metrics.three_way_agreement(
                [gemini_text, gt])["agreement_rate"],
            "gemini_vs_rapidocr": ag_rapid,
            "gemini_vs_bodhan": ag_bodhan,
            "valid_recall_vs_engines": vgs.get("valid_recall"),
            "readings": {"gemini": gemini_text, "rapidocr": rapid,
                         "bodhan": bodhan, "text_layer": gt},
        })
        print(f"  [{i}/{len(entries)}] {e['id'][:38]:<40} "
              f"vs rapid {ag_rapid if ag_rapid is None else round(ag_rapid, 2)} "
              f"vs bodhan {ag_bodhan if ag_bodhan is None else round(ag_bodhan, 2)}",
              flush=True)
        _write_payload(out_dir, payload)

    render_html(rows, os.path.join(out_dir, "audit.html"),
                engine_keys=("gemini", "rapidocr"),
                engine_labels=(f"{model} (GT candidate)", "RapidOCR (system under test)"),
                text_layer_key="bodhan",
                text_layer_label="bodhan reading (record only)")
    print(f"[gemini] {len(rows)} pages -> {verified_dir}")
    if payload["failed"]:
        print(f"[gemini] FAILED pages (rerun with --resume): "
              f"{[f['page'] for f in payload['failed']]}")
    print(f"[gemini] readings: {os.path.join(out_dir, 'gemini_readings.json')}")
    print(f"[gemini] audit: {os.path.join(out_dir, 'audit.html')}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Gemini transcription for the anchor")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "out", "anchor_gemini"))
    ap.add_argument("--readings", default=None,
                    help="local-engine readings JSON (for corroboration)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strips", type=int, default=2)
    ap.add_argument("--location", default=DEFAULT_LOCATION)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip pages whose verified/<page>.txt already exists")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-strip call timeout in seconds")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    run(args.data_dir, args.out, readings_path=args.readings, model=args.model,
        strips=args.strips, limit=args.limit, location=args.location,
        resume=args.resume, timeout_s=args.timeout, retries=args.retries)


if __name__ == "__main__":
    main()
