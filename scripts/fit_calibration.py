"""
fit_calibration.py — fit a temperature for OCR token confidence (per domain).

Confidence calibration: p_correct = sigmoid(logit(conf) / T), T fit by
minimizing BCE on a *development* set (never a frozen eval set). The fitted
file is committed under calibration/ and applied by document_ocr for ranking
and for the low-confidence flag; text is never changed.

Usage:
    python scripts/fit_calibration.py --data-dir data/doc_eval/heidata_dev --lang ne \
        --out calibration/rapidocr_devanagari_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402
import doc_metrics  # noqa: E402


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def collect_pairs(data_dir: str, lang: str) -> List[Tuple[float, bool]]:
    from document_ocr import ocr_page
    manifest = doc_data.load_dataset(data_dir)
    pairs: List[Tuple[float, bool]] = []
    for e in manifest["entries"]:
        img = doc_data.imread_safe(e["_clean_path"])
        if img is None:
            continue
        gt = open(e["_gt_path"], encoding="utf-8").read()
        gt_toks = {doc_metrics._norm_tok(t) for t in doc_metrics.tokens_of(gt)}
        res = ocr_page(img, backend="rapidocr", lang=lang)
        for t in res.tokens:
            pairs.append((t.conf, doc_metrics._norm_tok(t.text) in gt_toks))
    return pairs


def ece(probs, labels, bins: int = 10) -> float:
    probs, labels = np.asarray(probs), np.asarray(labels)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def _auc(scores, labels) -> float:
    """Rank AUC: P(score of a correct token > score of a wrong token)."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        for u_idx in np.where(counts > 1)[0]:
            m = inv == u_idx
            ranks[m] = ranks[m].mean()
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _pava(values, weights):
    """Pool-adjacent-violators: monotone non-decreasing fit of bin means."""
    vals: List[float] = []
    wts: List[float] = []
    for v, w in zip(values, weights):
        vals.append(float(v))
        wts.append(float(w))
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            w = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / w)
            wts.append(w)
    return vals, wts


def fit_isotonic(pairs: List[Tuple[float, bool]], n_bins: int = 20) -> Dict:
    """Monotone reliability mapping conf -> P(correct) (quantile bins + PAVA).

    Temperature scaling cannot fit this domain: probabilities are saturated
    near 1 while accuracy is 0.28, so the BCE optimum inverts ranking (T<0)
    instead of fixing the scale. Isotonic stays monotone, so the ranking
    (AUC) is preserved and the probability scale becomes honest.
    """
    confs = np.array([c / 100.0 for c, _ in pairs])
    labels = np.array([1.0 if ok else 0.0 for _, ok in pairs])
    order = np.argsort(confs, kind="mergesort")
    confs_s, labels_s = confs[order], labels[order]
    chunks = np.array_split(np.arange(len(confs_s)), n_bins)
    xs, means, weights = [], [], []
    for ch in chunks:
        if len(ch) == 0:
            continue
        xs.append(float(confs_s[ch].mean()))
        means.append(float(labels_s[ch].mean()))
        weights.append(float(len(ch)))
    mono_vals, mono_wts = _pava(means, weights)
    # piecewise-constant lookup: boundaries at the bin x-positions
    return {
        "x": [round(x, 6) for x in xs],
        "y": [round(y, 6) for y in mono_vals],
        "n_bins": n_bins,
    }


def apply_isotonic(conf_pct: float, mapping: Dict) -> float:
    from calibration import apply_isotonic as _apply
    return _apply(conf_pct, mapping)


def fit_temperature(pairs: List[Tuple[float, bool]],
                    grid=None) -> Dict:
    confs = np.array([c / 100.0 for c, _ in pairs])
    labels = np.array([1.0 if ok else 0.0 for _, ok in pairs])
    if grid is None:
        pos = np.concatenate([np.arange(0.25, 20.0, 0.25),
                              np.arange(20.0, 121.0, 1.0)])
        grid = np.concatenate([-pos[::-1], pos])
    best_T, best_nll = None, float("inf")
    for T in grid:
        p = 1.0 / (1.0 + np.exp(-_logit(confs) / T))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = float(nll), float(T)
    p_cal = 1.0 / (1.0 + np.exp(-_logit(confs) / best_T))
    auc = _auc(confs, labels)
    return {
        "temperature": best_T,
        "nll": round(best_nll, 5),
        "n_tokens": len(pairs),
        "accuracy": round(float(labels.mean()), 4),
        "auc_raw": round(auc, 4),
        "direction": ("normal" if auc >= 0.5 else "inverted"),
        "ece_raw": round(ece(confs, labels), 4),
        "ece_calibrated": round(ece(p_cal, labels), 4),
        "grid_max": float(abs(grid).max()),
        "at_grid_edge": bool(abs(abs(best_T) - abs(grid).max()) < 1e-9),
    }


def main():
    ap = argparse.ArgumentParser(description="Fit confidence temperature")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="rapidocr")
    args = ap.parse_args()

    t0 = time.time()
    pairs = collect_pairs(args.data_dir, args.lang)
    print(f"[calib] {len(pairs)} tokens from {args.data_dir} in {time.time()-t0:.0f}s")
    result = fit_temperature(pairs)
    isotonic = fit_isotonic(pairs)
    confs = np.array([c / 100.0 for c, _ in pairs])
    labels = np.array([1.0 if ok else 0.0 for _, ok in pairs])
    iso_probs = np.array([apply_isotonic(c, isotonic) / 100.0 for c, _ in pairs])
    ece_iso = round(ece(iso_probs, labels), 4)
    print(f"[calib] temperature T={result['temperature']} "
          f"(ECE {result['ece_raw']} -> {result['ece_calibrated']}, AUC {result['auc_raw']})")
    print(f"[calib] isotonic ECE {result['ece_raw']} -> {ece_iso}")
    if result["at_grid_edge"]:
        print("[calib] temperature hit the grid edge -> structurally unsuitable "
              "here (saturated probabilities); shipping the isotonic map")
    out = {
        "domain": os.path.basename(args.data_dir.rstrip("/\\")),
        "engine": args.engine,
        "lang": args.lang,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dev_manifest_sha256": hashlib.sha256(
            open(os.path.join(args.data_dir, "manifest.json"), "rb").read()).hexdigest(),
        "isotonic": isotonic,
        "ece_isotonic": ece_iso,
        "temperature_evidence": result,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
