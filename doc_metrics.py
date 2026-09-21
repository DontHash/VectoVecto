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


_DEVA_DIGIT_MAP = {chr(0x0966 + i): str(i) for i in range(10)}
_DIGIT_RUN_RE = None


def digit_string(text: str) -> str:
    """All digit runs of `text`, Devanagari digits normalized to ASCII.

    'रु. १२०.५०' -> '120.50'; 'total: 1,200.00' -> '1,200.00'. Only
    separators *inside* a number are kept, so sentence punctuation is not
    mistaken for an amount.
    """
    import re

    global _DIGIT_RUN_RE
    if _DIGIT_RUN_RE is None:
        _DIGIT_RUN_RE = re.compile(
            r"[0-9\u0966-\u096f](?:[0-9\u0966-\u096f.,:/-]*[0-9\u0966-\u096f])?")
    runs = _DIGIT_RUN_RE.findall(text)
    return " ".join("".join(_DEVA_DIGIT_MAP.get(c, c) for c in run)
                    for run in runs)


def digit_exact(gt: str, hyp: str) -> bool:
    """Exact whole-page digit-sequence match (the 'money is right' check)."""
    return digit_string(gt) == digit_string(hyp)


def bag_stats(gt: str, hyp: str) -> Dict[str, float]:
    """Bag-of-tokens match between GT and hypothesis (order-insensitive).

    Works without Token objects, so it covers recognizers that return plain
    text (generative models). `miss_rate`: share of GT tokens absent from the
    hypothesis (recall failure). `invented_rate`: share of hypothesis tokens
    absent from GT (hallucination proxy; on incomplete GT it is an upper
    bound).
    """
    gt_toks = {_norm_tok(t) for t in tokens_of(gt) if _norm_tok(t)}
    hyp_toks = [_norm_tok(t) for t in tokens_of(hyp) if _norm_tok(t)]
    hyp_set = set(hyp_toks)
    matched = len(gt_toks & hyp_set)
    invented = sum(1 for t in hyp_toks if t and t not in gt_toks)
    return {
        "gt_tokens": len(gt_toks),
        "hyp_tokens": len(hyp_toks),
        "matched": matched,
        "miss_rate": (1.0 - matched / len(gt_toks)) if gt_toks else 0.0,
        "invented_tokens": invented,
        "invented_rate": (invented / len(hyp_toks)) if hyp_toks else 0.0,
    }


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_match_report(gt_boxes: Sequence, hyp_boxes: Sequence,
                     iou_thresh: float = 0.5,
                     image_size: Sequence = None) -> Dict:
    """Greedy one-to-one IoU matching between GT and detected text boxes.

    `gt_boxes` / `hyp_boxes`: sequences of (x0, y0, x1, y1) or dicts with a
    "bbox" key. Returns GT coverage (share of annotated words/lines that got a
    detection on them) and the extra rate (detections matching no GT box,
    usually noise or split/merged annotations). This is the detection half of
    the pipeline that CER alone cannot see.

    When granularity differs (GT line boxes vs detected word boxes), IoU
    coverage under-counts splits/merges. With `image_size=(w, h)` a rasterized
    area view is added: a GT box counts as covered when >=50% of its area lies
    under the union of hyp boxes, and a hyp box counts as extra when its
    center lies outside every GT box. Both views are reported.
    """
    def _bbox(x):
        return tuple(x["bbox"]) if isinstance(x, dict) else tuple(x)

    gts = [_bbox(b) for b in gt_boxes]
    hyps = [_bbox(b) for b in hyp_boxes]
    pairs = []
    for gi, g in enumerate(gts):
        for hi, h in enumerate(hyps):
            v = _iou(g, h)
            if v >= iou_thresh:
                pairs.append((v, gi, hi))
    pairs.sort(key=lambda p: -p[0])
    used_g, used_h = set(), set()
    ious = []
    for v, gi, hi in pairs:
        if gi in used_g or hi in used_h:
            continue
        used_g.add(gi)
        used_h.add(hi)
        ious.append(v)
    matched = len(used_g)
    report = {
        "gt_total": len(gts),
        "hyp_total": len(hyps),
        "matched": matched,
        "coverage": matched / len(gts) if gts else None,
        "extra": len(hyps) - len(used_h),
        "extra_rate": (len(hyps) - len(used_h)) / len(hyps) if hyps else None,
        "mean_iou": float(sum(ious) / len(ious)) if ious else None,
        "iou_thresh": iou_thresh,
    }
    if image_size and gts and hyps:
        import numpy as np
        w, h = int(image_size[0]), int(image_size[1])
        scale = min(1.0, 800.0 / max(1, w, h))
        mw, mh = max(1, int(w * scale)), max(1, int(h * scale))
        mask = np.zeros((mh, mw), dtype=np.uint8)
        for x0, y0, x1, y1 in hyps:
            mask[max(0, int(y0 * scale)):max(0, int(y1 * scale)),
                 max(0, int(x0 * scale)):max(0, int(x1 * scale))] = 255
        covered = 0
        for x0, y0, x1, y1 in gts:
            gx0, gy0 = max(0, int(x0 * scale)), max(0, int(y0 * scale))
            gx1, gy1 = min(mw, int(x1 * scale)), min(mh, int(y1 * scale))
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            area = (gx1 - gx0) * (gy1 - gy0)
            inside = int((mask[gy0:gy1, gx0:gx1] > 0).sum())
            if inside / area >= 0.5:
                covered += 1
        outside = 0
        for x0, y0, x1, y1 in hyps:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if not any(gx0 <= cx <= gx1 and gy0 <= cy <= gy1
                       for gx0, gy0, gx1, gy1 in gts):
                outside += 1
        report["gt_area_coverage"] = covered / len(gts)
        report["hyp_center_extra_rate"] = outside / len(hyps)
    return report


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


def ece(tokens: Sequence[Token], gt_text: str, bins: int = 10,
        conf_attr: str = "conf") -> float:
    """Expected calibration error of token confidence vs exact-match correctness.

    `conf_attr` selects the confidence field ("conf" raw, "cal_conf"
    calibrated-by-isotonic). Tokens without the attribute fall back to raw.
    """
    gt_set = {_norm_tok(t) for t in tokens_of(gt_text)}
    if not tokens:
        return 0.0
    buckets = [[] for _ in range(bins)]
    for tok in tokens:
        raw = getattr(tok, conf_attr, None) or tok.conf
        p = max(0.0, min(1.0, raw / 100.0))
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
