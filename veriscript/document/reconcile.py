"""Page-level constraint reconciliation (docs/HARNESS_PLAN.md track B).

The model proposes (candidates from the re-pass, the audit stream, the
verifier and the multi-read panel); the page disposes. This module never
edits text: a candidate that page context prefers over the engine reading
becomes a `context_conflict` flag (weight 2.0) plus a `suggestions` entry
with a human-readable `why`. `alt_text` is never written.

Constraints (v1, page-local, no external tables):

  1. date-column agreement - date-shaped tokens in one x-cluster share their
     year/month; a token whose reading deviates and whose candidate set
     contains exactly one reading matching the majority is flagged;
  2. format consistency - among tied candidates, one whose separator pattern
     matches the cluster majority wins (tie-breaker only);
  3. prefix groups - same-shape number tokens in one column share their first
     digit group; a single-candidate first-group correction is flagged;
  4. in-page repeat - among tied candidates, a reading that already occurs
     elsewhere on the page wins (tie-breaker only).

Ties that survive the tie-breakers are never flagged: ambiguity is not
evidence. Sums/checksum constraints are explicitly a later version.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from veriscript.document.ocr import (Token, _DATE_BARE_RE, _DATE_SEP_RE,
                                     _DEVA_TO_LATIN, _digits_of)

CONTEXT_FLAG = "context_conflict"
MIN_CLUSTER = 4
MAJORITY = 0.7
PREFIX_MAJORITY = 0.75
MAX_SUGGESTIONS = 3


def _date_components(text: str) -> Optional[Tuple[int, int, int]]:
    """(year, month, day) from the first date-shaped run, or None."""
    for m in _DATE_SEP_RE.finditer(text or ""):
        return tuple(int(g.translate(_DEVA_TO_LATIN)) for g in m.groups())
    for m in _DATE_BARE_RE.finditer(text or ""):
        s = m.group(1).translate(_DEVA_TO_LATIN)
        return int(s[:4]), int(s[4:6]), int(s[6:8])
    return None


def _digit_groups(text: str) -> List[str]:
    return re.findall(r"[०-९\d]+", text or "")


def _sep_pattern(text: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"[^\s०-९\dA-Za-z]+", text or ""))


def _candidates(tok: Token) -> List[str]:
    """Distinct alternative readings whose digit strings differ from base."""
    base = _digits_of(tok.text)
    out: List[str] = []
    seen = {base}
    texts = [tok.repass_text, tok.alt_text]
    texts += [r.get("text") for r in tok.reads]
    texts += [s.get("text") for s in tok.suggestions]
    for text in texts:
        if not text:
            continue
        digits = _digits_of(text)
        if digits and digits not in seen:
            seen.add(digits)
            out.append(str(text).strip())
    return out


def _x_clusters(tokens: Sequence[Token],
                min_size: int = MIN_CLUSTER) -> List[List[Token]]:
    """Greedy x-overlap clusters (>= 50% of the narrower box)."""
    clusters: List[List[Token]] = []
    for tok in sorted(tokens, key=lambda t: (t.bbox[0], t.bbox[1])):
        x0, x1 = tok.bbox[0], tok.bbox[2]
        for cluster in clusters:
            cx0 = min(t.bbox[0] for t in cluster)
            cx1 = max(t.bbox[2] for t in cluster)
            overlap = max(0, min(x1, cx1) - max(x0, cx0))
            if overlap / max(1, min(x1 - x0, cx1 - cx0)) >= 0.5:
                cluster.append(tok)
                break
        else:
            clusters.append([tok])
    return [c for c in clusters if len(c) >= min_size]


def _majority(values: Sequence, bar: float) -> Optional[object]:
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return value if count >= bar * len(values) else None


def _tie_break(picks: List[str], cluster: Sequence[Token],
               all_tokens: Sequence[Token]) -> Optional[str]:
    """Constraints 2/4: separator majority, then in-page repeat."""
    if len(picks) == 1:
        return picks[0]
    seps = Counter(_sep_pattern(t.text) for t in cluster)
    if seps:
        sep, count = seps.most_common(1)[0]
        if count >= MAJORITY * len(cluster):
            matching = [p for p in picks if _sep_pattern(p) == sep]
            if len(matching) == 1:
                return matching[0]
    counts = Counter(_digits_of(t.text) for t in all_tokens)
    scored = [(counts.get(_digits_of(p), 0), p) for p in picks]
    best = max(scored)
    if best[0] >= 2 and sum(1 for s, _ in scored if s == best[0]) == 1:
        return best[1]
    return None


def _flag(tok: Token, candidate: str, why: str) -> bool:
    """Add the flag + suggestion once; never edits text/alt_text."""
    text = candidate.strip()
    if not text or _digits_of(text) == _digits_of(tok.text):
        return False
    added = False
    if CONTEXT_FLAG not in tok.flags:
        tok.flags.append(CONTEXT_FLAG)
        added = True
    if not any(s.get("text") == text and s.get("source") == "context"
               for s in tok.suggestions):
        if len(tok.suggestions) < MAX_SUGGESTIONS:
            tok.suggestions.append({"text": text, "source": "context",
                                    "why": why})
    return added


def _reconcile_dates(cluster: List[Token],
                     all_tokens: Sequence[Token]) -> int:
    parts = [_date_components(t.text) for t in cluster]
    maj_year = _majority([p[0] for p in parts if p], MAJORITY)
    maj_month = _majority([p[1] for p in parts if p], MAJORITY)
    if maj_year is None and maj_month is None:
        return 0
    flagged = 0
    for tok, base in zip(cluster, parts):
        if base is None:
            continue
        candidates = [(_date_components(c), c) for c in _candidates(tok)]
        candidates = [(p, c) for p, c in candidates if p is not None]
        picks: List[str] = []
        if maj_year is not None and base[0] != maj_year:
            picks = [c for p, c in candidates if p[0] == maj_year]
        elif maj_month is not None and base[1] != maj_month:
            picks = [c for p, c in candidates if p[1] == maj_month]
        if not picks:
            continue
        pick = _tie_break(picks, cluster, all_tokens)
        if pick is None:
            continue
        why = f"date column: {len(cluster)} tokens share year/month"
        if _flag(tok, pick, why):
            flagged += 1
    return flagged


def _reconcile_prefixes(digit_tokens: Sequence[Token],
                        all_tokens: Sequence[Token]) -> int:
    by_shape: Dict[Tuple[int, ...], List[Token]] = {}
    for tok in digit_tokens:
        if _date_components(tok.text) is not None:
            continue  # the date constraint owns date shapes
        groups = _digit_groups(tok.text)
        if len(groups) < 2:
            continue
        by_shape.setdefault(tuple(len(g) for g in groups), []).append(tok)
    flagged = 0
    for toks in by_shape.values():
        for cluster in _x_clusters(toks):
            groups_by_tok = {id(t): _digit_groups(t.text) for t in cluster}
            maj = _majority([g[0] for g in groups_by_tok.values()],
                            PREFIX_MAJORITY)
            if maj is None:
                continue
            for tok in cluster:
                groups = groups_by_tok[id(tok)]
                if groups[0] == maj:
                    continue
                picks = []
                for c in _candidates(tok):
                    cg = _digit_groups(c)
                    if (len(cg) == len(groups) and cg[0] == maj
                            and cg[1:] == groups[1:]):
                        picks.append(c)
                if not picks:
                    continue
                pick = _tie_break(picks, cluster, all_tokens)
                if pick is None:
                    continue
                if _flag(tok, pick, f"prefix column: {maj} majority"):
                    flagged += 1
    return flagged


def reconcile_page(tokens: Sequence[Token]) -> Dict:
    """Flag tokens whose candidates agree with page-level constraints.

    Pure post-OCR pass: deterministic, text never changes. Returns counters
    for `meta["reconcile"]`.
    """
    digit_tokens = [t for t in tokens if t.has_digits]
    stats = {
        "candidates": sum(1 for t in digit_tokens if _candidates(t)),
        "date_flags": 0,
        "prefix_flags": 0,
        "flags": 0,
    }
    dates = [t for t in digit_tokens if _date_components(t.text)]
    date_clusters = _x_clusters(dates)
    stats["clusters"] = len(date_clusters)
    for cluster in date_clusters:
        stats["date_flags"] += _reconcile_dates(cluster, tokens)
    stats["prefix_flags"] = _reconcile_prefixes(digit_tokens, tokens)
    stats["flags"] = sum(1 for t in tokens if CONTEXT_FLAG in t.flags)
    return stats
