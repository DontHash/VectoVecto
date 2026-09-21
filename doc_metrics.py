"""
doc_metrics.py — honesty metrics for the document pipeline.

All metrics are computed on normalized text (see doc_data.normalize_text):
CER/WER via jiwer; hallucination and coverage at token level; ECE for
calibration of OCR confidence.

Definitions:
  * hallucination : a token in the *restored* OCR output that appears neither in
                    the ground truth nor in the *raw* OCR output. The restore
                    step invented it. Tracked separately for digit tokens.
  * coverage      : fraction of true token errors that were flagged
                    (low conf / digit uncertain). Higher = more honest UX.
  * false alarm   : fraction of flagged tokens that were actually correct.
                    Bounds how annoying flagging gets.
  * ECE           : expected calibration error of confidence vs exact-match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import jiwer

from document_ocr import Token

_PUNCT = ".,;:!?()[]{}\"'`“”‘’«»·-–—/\\|@#$%^&*+=~<>"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def tokens_of(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", normalize_text(text)) if t]


def _norm_tok(tok: str) -> str:
    return tok.strip(_PUNCT).lower()


def cer(gt: str, hyp: str) -> float:
    gt, hyp = normalize_text(gt), normalize_text(hyp)
    if not gt:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(gt, hyp))


def wer(gt: str, hyp: str) -> float:
    gt, hyp = normalize_text(gt), normalize_text(hyp)
    if not gt:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(gt, hyp))


def cer_bag(gt: str, hyp: str) -> float:
    """Order-insensitive CER: sort tokens alphabetically before comparing.

    Use when the ground-truth order is unreliable (word-level annotations,
    receipts). 'total 9.00 tax 1.20' vs 'tax 1.20 total 9.00' scores 0 here
    but would score like a total miss with plain character CER.
    """
    gt_bag = " ".join(sorted(tokens_of(gt)))
    hyp_bag = " ".join(sorted(tokens_of(hyp)))
    if not gt_bag:
        return 0.0 if not hyp_bag else 1.0
    return float(jiwer.cer(gt_bag, hyp_bag))


def digit_tokens(text: str) -> List[str]:
    return [t for t in tokens_of(text) if any(ch.isdigit() for ch in t)]


def digit_cer(gt: str, hyp: str, bag: bool = False) -> float:
    """CER over digit-bearing tokens only (amounts, dates, invoice numbers).

    This is the 'money metric': a 0.01 vs 0.07 misread changes the total.
    """
    gt_toks = digit_tokens(gt)
    hyp_toks = digit_tokens(hyp)
    if bag:
        gt_toks, hyp_toks = sorted(gt_toks), sorted(hyp_toks)
    gt_d = " ".join(gt_toks)
    hyp_d = " ".join(hyp_toks)
    if not gt_d:
        return 0.0 if not hyp_d else 1.0
    return float(jiwer.cer(gt_d, hyp_d))


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> List[float]:
    """Percentile bootstrap CI of the mean. Deterministic for a given seed.

    Small eval sets (30-60 pages) make point deltas meaningless without an
    error bar; this is the cheapest honest one. Returns [lo, hi] at
    (alpha/2, 1-alpha/2).
    """
    import numpy as np

    vals = np.asarray([v for v in values if v == v and v is not None], dtype=float)
    if vals.size == 0:
        return [float("nan"), float("nan")]
    if vals.size == 1:
        return [float(vals[0]), float(vals[0])]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return [float(lo), float(hi)]


@dataclass
class TokenStats:
    total: int = 0
    digit_total: int = 0
    errors: int = 0
    flagged: int = 0
    flagged_errors: int = 0
    flagged_correct: int = 0
    digit_errors: int = 0
    digit_flagged_errors: int = 0
    digit_flagged: int = 0  # flagged digit-bearing tokens (errors + correct)

    @property
    def coverage(self) -> float:
        return self.flagged_errors / self.errors if self.errors else 1.0

    @property
    def false_alarm_rate(self) -> float:
        return self.flagged_correct / self.flagged if self.flagged else 0.0

    @property
    def digit_coverage(self) -> float:
        """Of the wrong digit tokens, how many raised a flag. The money bar."""
        return (self.digit_flagged_errors / self.digit_errors
                if self.digit_errors else 1.0)

    @property
    def digit_false_alarm_rate(self) -> float:
        correct = self.digit_flagged - self.digit_flagged_errors
        return correct / self.digit_flagged if self.digit_flagged else 0.0

    @property
    def token_error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0


def token_stats(tokens: Sequence[Token], gt_text: str) -> TokenStats:
    """Exact-match token accuracy + flagging quality. Use *word* granularity
    (Tesseract) or split line tokens into words for RapidOCR lines."""
    gt_set = {_norm_tok(t) for t in tokens_of(gt_text)}
    stats = TokenStats()
    for tok in tokens:
        parts = tokens_of(tok.text)
        if tok.granularity == "line" and len(parts) > 1:
            ok = all(_norm_tok(p) in gt_set for p in parts)
            n = len(parts)
        else:
            parts = [tok.text]
            ok = _norm_tok(tok.text) in gt_set
            n = 1
        has_digit = any(any(ch.isdigit() for ch in p) for p in parts)
        stats.total += n
        stats.digit_total += n if has_digit else 0
        flagged = bool(tok.flags)
        stats.flagged += n if flagged else 0
        stats.digit_flagged += n if (flagged and has_digit) else 0
        if not ok:
            stats.errors += n
            stats.flagged_errors += n if flagged else 0
            if has_digit:
                stats.digit_errors += n
                stats.digit_flagged_errors += n if flagged else 0
        else:
            stats.flagged_correct += n if flagged else 0
    return stats


def hallucination_report(gt_text: str, raw_text: str, restored_text: str,
                         extra_refs: Sequence[str] = ()) -> Dict[str, float]:
    """Tokens present in restored but neither in GT nor in raw.

    For real-photo sets the GT annotation is incomplete, so a token no method
    but this one produced is more likely a genuine invention — pass the other
    methods' texts as `extra_refs` there (peer-confirmed text is not invented).
    On synthetic sets, where GT is exact, leave extra_refs empty.
    """
    gt = {_norm_tok(t) for t in tokens_of(gt_text)}
    raw = {_norm_tok(t) for t in tokens_of(raw_text)}
    for ref in extra_refs:
        raw |= {_norm_tok(t) for t in tokens_of(ref)}
    rest = tokens_of(restored_text)
    invented = [t for t in rest if _norm_tok(t) not in gt and _norm_tok(t) not in raw]
    invented_digits = [t for t in invented if any(ch.isdigit() for ch in t)]
    n = max(len(rest), 1)
    return {
        "invented": len(invented),
        "invented_rate": len(invented) / n,
        "invented_digits": len(invented_digits),
        "invented_digit_rate": len(invented_digits) / n,
        "tokens": len(rest),
    }


def ece(tokens: Sequence[Token], gt_text: str, bins: int = 10) -> float:
    """Expected calibration error of token confidence vs exact-match correctness."""
    gt_set = {_norm_tok(t) for t in tokens_of(gt_text)}
    if not tokens:
        return 0.0
    buckets = [[] for _ in range(bins)]
    for tok in tokens:
        p = max(0.0, min(1.0, tok.conf / 100.0))
        correct = 1.0 if _norm_tok(tok.text) in gt_set else 0.0
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((p, correct))
    n = len(tokens)
    err = 0.0
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(c for _, c in b) / len(b)
        err += (len(b) / n) * abs(conf - acc)
    return err


if __name__ == "__main__":
    from document_ocr import Token as T

    gt = "INVOICE 1200.00 Total Amount"
    hyp = "INVOICE 1200.00 Total Am0unt"
    print(f"CER {cer(gt, hyp):.4f}  WER {wer(gt, hyp):.4f}")
    toks = [T(text="INVOICE", conf=99, bbox=(0, 0, 1, 1), granularity="word"),
            T(text="1200.00", conf=55, bbox=(0, 0, 1, 1), granularity="word",
              flags=["low_conf"]),
            T(text="Am0unt", conf=90, bbox=(0, 0, 1, 1), granularity="word")]
    st = token_stats(toks, gt)
    print(f"stats: errors={st.errors} flagged={st.flagged} coverage={st.coverage:.2f} "
          f"false_alarm={st.false_alarm_rate:.2f} ECE={ece(toks, gt):.3f}")
    hr = hallucination_report(gt, "INVOICE 1200.00 Total Am0unt", "INVOICE 1200.00 Total 4m0unt")
    print("hallucination:", hr)
    assert st.errors == 1 and st.coverage == 0.0 and st.false_alarm_rate == 1.0
    assert hr["invented"] == 1
    print("doc_metrics OK")
