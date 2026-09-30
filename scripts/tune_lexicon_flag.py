"""
tune_lexicon_flag.py — dev-set sweep for the `unknown_word` flag (W-B).

Runs RapidOCR once over a *development* set (never a frozen eval set), then
re-scores the review queue for candidate (oov_frac, risk-weight) pairs by
re-applying only the lexicon flag to copies of the tokens. The shipped
constants (`document_ocr.LEXICON_OOV_FRAC`, `RISK_WEIGHTS["unknown_word"]`)
are the row picked here; the frozen confirmation is run separately with
`evals/harness/eval_flags.py`.

Usage:
    python scripts/tune_lexicon_flag.py --data-dir data/doc_eval/heidata_dev \
        --json out/lexicon_tune_dev.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.core import data as doc_data  # noqa: E402
from veriscript.core import metrics as doc_metrics  # noqa: E402
from veriscript.document import ocr as document_ocr  # noqa: E402
from veriscript.document.ocr import apply_unknown_word, ocr_page  # noqa: E402
from veriscript.lexicon import load_lexicon  # noqa: E402

DEVA_DIGITS = {chr(0x0966 + i) for i in range(10)}
TOPK = (5, 10)


def _digit_tokens(text: str) -> List[str]:
    return [t for t in doc_metrics.tokens_of(text)
            if any(c in DEVA_DIGITS or c.isdigit() for c in t)]


def _page_gt(entry: Dict) -> str:
    with open(entry["_gt_path"], encoding="utf-8") as f:
        return f.read()


def collect_pages(data_dir: str, lang: str, limit: int,
                  repass_digits: bool) -> List[Tuple[List, str]]:
    """OCR every dev page once; tokens keep their non-lexicon flags."""
    manifest = doc_data.load_dataset(data_dir)
    entries = manifest["entries"][:limit] if limit else manifest["entries"]
    pages = []
    for i, e in enumerate(entries, 1):
        img = doc_data.imread_safe(e["_degraded_path"])
        if img is None:
            continue
        rec = ocr_page(img, backend="rapidocr", lang=lang,
                       recheck_digits=repass_digits)
        for tok in rec.tokens:
            while "unknown_word" in tok.flags:
                tok.flags.remove("unknown_word")
        pages.append((rec.tokens, _page_gt(e)))
        print(f"  [{i}/{len(entries)}] {e['id']}: {len(rec.tokens)} tokens",
              flush=True)
    return pages


def score(pages: List[Tuple[List, str]]) -> Dict:
    """Aggregated queue metrics (same definitions as eval_flags.py)."""
    t_hits = {k: 0 for k in TOPK}
    t_top = {k: 0 for k in TOPK}
    t_total = 0
    d_hits = {k: 0 for k in TOPK}
    d_top = {k: 0 for k in TOPK}
    d_total = 0
    flagged = 0
    any_flagged = 0
    for tokens, gt in pages:
        gt_digits = {doc_metrics.digit_string(t) for t in _digit_tokens(gt)}
        gt_digits.discard("")
        errs = []
        for t in tokens:
            if not t.has_digits:
                continue
            d = doc_metrics.digit_string(t.text)
            if d and d not in gt_digits:
                errs.append(t)
        d_total += len(errs)
        err_ids = {id(t) for t in errs}
        queue = document_ocr.review_queue(tokens)
        for k in TOPK:
            top = queue[:k]
            d_hits[k] += sum(1 for t in top if id(t) in err_ids)
            d_top[k] += len(top)
        tq = doc_metrics.queue_stats(tokens, gt, top_k=TOPK)
        for k in TOPK:
            t_hits[k] += tq[f"hits@{k}"]
            t_top[k] += min(k, tq["queue"])
        t_total += tq["errors"]
        flagged += sum(1 for t in tokens if "unknown_word" in t.flags)
        any_flagged += sum(1 for t in tokens if t.flags)

    def safe(num, den):
        return round(num / den, 4) if den else None

    return {
        "tokens": sum(len(toks) for toks, _ in pages),
        "pages": len(pages),
        "unknown_word_tokens": flagged,
        "flagged_tokens": any_flagged,
        "token_errors": t_total,
        **{f"token_recall@{k}": safe(t_hits[k], t_total) for k in TOPK},
        **{f"token_precision@{k}": safe(t_hits[k], t_top[k]) for k in TOPK},
        "digit_errors": d_total,
        **{f"digit_recall@{k}": safe(d_hits[k], d_total) for k in TOPK},
        **{f"digit_precision@{k}": safe(d_hits[k], d_top[k]) for k in TOPK},
    }


def sweep(pages: List[Tuple[List, str]], lexicon: set,
          fracs: List[float], weights: List[float]) -> List[Dict]:
    base = score(pages)
    rows = [{"oov_frac": None, "weight": None, "label": "baseline", **base}]
    for frac in fracs:
        for weight in weights:
            variants = copy.deepcopy(pages)
            document_ocr.RISK_WEIGHTS["unknown_word"] = weight
            for tokens, _gt in variants:
                apply_unknown_word(tokens, lexicon, oov_frac=frac)
            rows.append({"oov_frac": frac, "weight": weight,
                         "label": f"frac={frac} w={weight}",
                         **score(variants)})
    return rows


def pick(rows: List[Dict]) -> Dict:
    """Near-best token recall@10, then the safest digit queue.

    Recall differences below 0.5pp are treated as ties (dev resolution), and
    ties prefer digit R@10, then digit P@10, then the higher `oov_frac` (fewer
    flagged tokens at equal measured quality). This keeps the product's digit
    priority first: no config may cost more than 0.5pp of token precision or
    digit quality.
    """
    base = rows[0]
    ok = [r for r in rows[1:]
          if (r["token_precision@10"] or 0) >= (base["token_precision@10"] or 0) - 0.005
          and (r["digit_recall@10"] or 0) >= (base["digit_recall@10"] or 0) - 0.005
          and (r["digit_precision@10"] or 0) >= (base["digit_precision@10"] or 0) - 0.005]
    pool = ok or rows[1:]
    best_tok = max(r["token_recall@10"] or 0 for r in pool)
    near = [r for r in pool
            if (r["token_recall@10"] or 0) >= best_tok - 0.005]
    return max(near, key=lambda r: (r["digit_recall@10"] or 0,
                                    r["digit_precision@10"] or 0,
                                    r["token_recall@10"] or 0,
                                    r["token_precision@10"] or 0,
                                    r["oov_frac"] or 0))


def main():
    ap = argparse.ArgumentParser(description="Tune the unknown_word flag")
    ap.add_argument("--data-dir", default=os.path.join(
        BASE_DIR, "data", "doc_eval", "heidata_dev"))
    ap.add_argument("--lang", default="ne")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fracs", default="0.0,0.34,0.5,0.67")
    ap.add_argument("--weights", default="0.5,1.0,1.5")
    ap.add_argument("--repass-digits", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="mirror the shipped Devanagari default (2x re-pass)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    lex = load_lexicon()
    if lex is None:
        raise SystemExit("no lexicon - run scripts/fetch_nepali_lexicon.py "
                         "first (or set VERISCRIPT_LEXICON)")
    print(f"[tune] lexicon {lex['source']}: {lex['count']} words")
    pages = collect_pages(args.data_dir, args.lang, args.limit,
                          args.repass_digits)
    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    rows = sweep(pages, lex["words"], fracs, weights)

    print("\n=== unknown_word dev sweep (heidata_dev) ===")
    header = ("config", "tokR@10", "tokP@10", "digR@10", "digP@10",
              "oov_toks")
    print("  " + "  ".join(f"{h:<12}" for h in header))
    for r in rows:
        print("  " + "  ".join(f"{str(v):<12}" for v in (
            r["label"], r["token_recall@10"], r["token_precision@10"],
            r["digit_recall@10"], r["digit_precision@10"],
            r["unknown_word_tokens"])))
    best = pick(rows)
    print(f"\n[tune] pick: {best['label']} "
          f"(token R@10 {best['token_recall@10']} / P@10 "
          f"{best['token_precision@10']})")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"data_dir": args.data_dir, "lang": args.lang,
                       "repass_digits": args.repass_digits,
                       "rows": rows, "pick": best}, f, indent=2)
        print(f"[tune] wrote {args.json}")


if __name__ == "__main__":
    main()
