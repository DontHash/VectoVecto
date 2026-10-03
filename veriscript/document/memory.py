"""Local correction memory (docs/HARNESS_PLAN.md track D2).

Text-only, opt-in, local: records accepted human corrections so repeated
values (dates, case numbers, amounts) can be suggested on later runs. Never
stores page images, token crops, bboxes or run ids. Disabled by default -
`VERISCRIPT_MEMORY=1` enables it, `VERISCRIPT_MEMORY_DIR` overrides the path.
Multi-tenant deployments keep it off so no visitor's corrections can leak into
another visitor's suggestions.

The store is append-only JSONL:

  {"kind": "correction", "key": "<flags signature>", "original": ...,
   "corrected": ..., "flags": [...], "digits": "20810427", "shape": [4,2,2],
   "engine": ..., "lang": ..., "created": ...}
  {"kind": "value", "value": "00962", "display": "०-०९६२",
   "corrected": "०९६२", "created": ...}
  {"kind": "accept"|"reject", "key": <flags signature>,
   "digits": <digits of the original reading>, "corrected": <suggested text>,
   "created": ...}
  {"kind": "value_accept"|"value_reject", "value": ..., "corrected": ...}

Suggestion matching: line level - same flags signature, exact full reading
first, else same skeleton and Hamming <= 1; value level - a repeated wrong
digit run (>= 4 digits, one edit from its correction) is substituted in place.
Line-level is preferred when both exist. A record whose feedback is
net-negative is suppressed. Suggestions are never auto-applied: the human
confirms and the web layer records the accept/reject.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

MEMORY_ENV = "VERISCRIPT_MEMORY"
DIR_ENV = "VERISCRIPT_MEMORY_DIR"
_TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".veriscript")
DEFAULT_PATH = os.path.join(DEFAULT_DIR, "memory.jsonl")
MAX_EVENTS = 20000
SUGGEST_LIMIT = 2
NEAR_HAMMING = 1
ACCEPT_FLOOR = 0.5
VALUE_MIN_DIGITS = 4      # value-level memory only for distinctive runs
VALUE_NEAR_HAMMING = 1

_RUN_RE = re.compile(
    r"[0-9\u0966-\u096f](?:[0-9\u0966-\u096f.,:/-]*[0-9\u0966-\u096f])?")


def memory_enabled() -> bool:
    """True only when the user explicitly opted in (env, checked per call)."""
    return os.environ.get(MEMORY_ENV, "").strip().lower() in _TRUTHY


def memory_path() -> str:
    return os.environ.get(DIR_ENV) or DEFAULT_PATH


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flags_key(flags: Sequence[str]) -> str:
    return "|".join(sorted({str(f) for f in (flags or [])}))


def _skeleton(text: str) -> str:
    """The text with every digit replaced by `#` (near-match shape)."""
    masked = re.sub(r"[०-९\d]", "#", text or "")
    return re.sub(r"\s+", " ", masked.strip()).casefold()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _digit_string(text: str) -> str:
    from veriscript.core.metrics import digit_string
    return digit_string(text or "")


def _digits(text: str) -> str:
    return "".join(c for c in _digit_string(text) if c.isdigit())


def _shape(text: str) -> List[int]:
    runs = _digit_string(text).split(" ")
    return [len("".join(c for c in r if c.isdigit())) for r in runs if r]


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _runs_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end())
            for m in _RUN_RE.finditer(text or "")]


def _edit_distance_le1(a: str, b: str) -> bool:
    """True when one insertion/deletion/substitution turns `a` into `b`."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return _hamming(a, b) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def _value_pairs(original: str, corrected: str) -> List[Tuple[str, str, str]]:
    """(digits key, original display, corrected display) for aligned runs.

    Pairs positionally only when both texts have the same number of digit
    runs and the digits differ by at most one edit - a plausible misreading,
    not a different number. Conservative on purpose: a value pair that is not
    near-identical is not memory, it is a different fact.
    """
    runs_original = _runs_with_spans(original)
    runs_corrected = _runs_with_spans(corrected)
    if not runs_original or len(runs_original) != len(runs_corrected):
        return []
    pairs: List[Tuple[str, str, str]] = []
    for (display_a, _sa, _ea), (display_b, _sb, _eb) in zip(runs_original,
                                                            runs_corrected):
        if display_a == display_b:
            continue
        digits_a, digits_b = _digits(display_a), _digits(display_b)
        if len(digits_a) < VALUE_MIN_DIGITS or not digits_b:
            continue
        if _edit_distance_le1(digits_a, digits_b):
            pairs.append((digits_a, display_a, display_b))
    return pairs


def _value_index(events: Sequence[Dict]) -> Dict[str, Dict[str, Dict]]:
    """digits key -> corrected display -> {text, count, accepted, rejected}."""
    index: Dict[str, Dict[str, Dict]] = {}
    for event in events:
        kind = event.get("kind")
        if kind == "value":
            value, corrected = event.get("value"), event.get("corrected")
            if not value or not corrected:
                continue
            rec = index.setdefault(str(value), {}).setdefault(str(corrected), {
                "text": str(corrected), "count": 0, "accepted": 0,
                "rejected": 0})
            rec["count"] += 1
        elif kind in ("value_accept", "value_reject"):
            value, corrected = event.get("value"), event.get("corrected")
            rec = index.get(str(value), {}).get(str(corrected))
            if rec is not None:
                rec["accepted" if kind == "value_accept" else "rejected"] += 1
    return index


def _value_match(index: Dict[str, Dict[str, Dict]],
                 digits: str) -> Optional[Dict]:
    bucket = index.get(digits)
    if bucket:
        usable = [rec for rec in bucket.values() if _usable(rec)]
        if usable:
            return max(usable, key=lambda rec: rec["count"])
    for value, candidates in index.items():
        if (len(value) == len(digits)
                and _hamming(value, digits) <= VALUE_NEAR_HAMMING):
            usable = [rec for rec in candidates.values() if _usable(rec)]
            if usable:
                return max(usable, key=lambda rec: rec["count"])
    return None


def _value_suggest(index: Dict[str, Dict[str, Dict]],
                   text: str) -> Optional[Dict]:
    """Substitute the best-supported wrong value in `text`, if any."""
    best: Optional[Tuple[int, Dict]] = None
    for run, start, end in _runs_with_spans(text):
        digits = _digits(run)
        if len(digits) < VALUE_MIN_DIGITS:
            continue
        rec = _value_match(index, digits)
        if rec is None or rec["text"] == run:
            continue
        suggestion = {
            "text": text[:start] + rec["text"] + text[end:],
            "source": "memory:value",
            "why": f"value {run} corrected {rec['count']}x before",
        }
        if best is None or rec["count"] > best[0]:
            best = (rec["count"], suggestion)
    return best[1] if best else None


def _iter_events(path: Optional[str] = None) -> List[Dict]:
    p = path or memory_path()
    if not os.path.isfile(p):
        return []
    events: List[Dict] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return events


def _append(event: Dict, path: Optional[str] = None) -> None:
    p = path or memory_path()
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def _trim(path: Optional[str] = None) -> None:
    p = path or memory_path()
    try:
        with open(p, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except OSError:
        return
    if len(lines) <= MAX_EVENTS:
        return
    keep = lines[-MAX_EVENTS:]
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(keep)


def _index(events: Sequence[Dict]) -> Dict[str, Dict[str, Dict]]:
    """flags signature -> original digits -> {text, count, accepted, rejected}."""
    index: Dict[str, Dict[str, Dict]] = {}
    for event in events:
        kind = event.get("kind")
        if kind == "correction":
            key, corrected = event.get("key"), event.get("corrected")
            if not key or not corrected:
                continue
            digits = event.get("digits") or ""
            rec = index.setdefault(key, {}).setdefault(digits, {
                "text": str(corrected), "count": 0, "accepted": 0,
                "rejected": 0})
            rec["text"] = str(corrected)  # the latest accepted fix wins
            rec["original"] = str(event.get("original") or "")
            rec["norm"] = _norm(event.get("original") or "")
            rec["count"] += 1
        elif kind in ("accept", "reject"):
            key = event.get("key")
            digits = event.get("digits") or ""
            rec = index.get(key, {}).get(digits)
            if rec is not None:
                rec["accepted" if kind == "accept" else "rejected"] += 1
    return index


def _usable(rec: Dict) -> bool:
    accepted, rejected = rec["accepted"], rec["rejected"]
    if rejected > accepted:
        return False
    if accepted + rejected >= 2 and accepted / (accepted + rejected) < ACCEPT_FLOOR:
        return False
    return True


def _lookup(index: Dict[str, Dict[str, Dict]], text: str,
            flags: Sequence[str]) -> Optional[Dict]:
    bucket = index.get(_flags_key(flags))
    if not bucket:
        return None
    digits = _digits(text)
    norm = _norm(text)
    rec = bucket.get(digits)
    if rec is not None and (rec.get("norm") != norm or not _usable(rec)):
        rec = None
    if rec is None:
        skeleton = _skeleton(text)
        best: Optional[Dict] = None
        for candidate_digits, candidate in bucket.items():
            if (candidate_digits and len(candidate_digits) == len(digits)
                    and _hamming(candidate_digits, digits) <= NEAR_HAMMING
                    and _skeleton(candidate.get("original", "")) == skeleton
                    and _usable(candidate)):
                if best is None or candidate["count"] > best["count"]:
                    best = candidate
        rec = best
    if rec is None:
        return None
    why = (f"corrected {rec['count']}x before" if rec["count"] > 1
           else "corrected before")
    if rec["accepted"]:
        why += f" · accepted {rec['accepted']}x"
    return {"text": rec["text"], "source": "memory", "why": why}


def suggest(text: str, flags: Sequence[str],
            path: Optional[str] = None) -> Optional[Dict]:
    """One suggestion for a flagged reading, or None. Never auto-applied."""
    if not memory_enabled():
        return None
    return _lookup(_index(_iter_events(path)), text, flags)


def augment_review(review: List[Dict], path: Optional[str] = None) -> int:
    """Append memory suggestions to review rows in place. Returns rows gained.

    Line-level (exact reading) first; when there is none, a value-level
    suggestion substitutes the best-supported wrong digit run.
    """
    if not memory_enabled() or not review:
        return 0
    events = _iter_events(path)
    index = _index(events)
    values = _value_index(events)
    added = 0
    for item in review:
        if not isinstance(item, dict):
            continue
        match = _lookup(index, item.get("text") or "", item.get("flags") or [])
        if match is None:
            match = _value_suggest(values, item.get("text") or "")
        if match is None:
            continue
        suggestions = item.setdefault("suggestions", [])
        if any(str(s.get("text")) == match["text"] for s in suggestions):
            continue
        if len(suggestions) >= SUGGEST_LIMIT:
            continue
        suggestions.append(match)
        added += 1
    return added


def suggest_value(text: str, path: Optional[str] = None) -> Optional[Dict]:
    """A value-level suggestion (one wrong digit run substituted), or None."""
    if not memory_enabled():
        return None
    return _value_suggest(_value_index(_iter_events(path)), text)


def record_corrections(entries: Sequence[Dict],
                       snapshots: Dict[int, Tuple[str, Sequence[str]]],
                       engine: str = "", lang: str = "",
                       path: Optional[str] = None) -> int:
    """Persist accepted corrections and suggestion feedback.

    `entries` are the raw correction dicts from the API (index, corrected,
    action, and optionally `suggested`); `snapshots` maps a token index to
    its pre-correction `(text, flags)`. Text-only: bbox and run ids are
    deliberately not stored. Returns the number of correction records written.
    """
    if not memory_enabled():
        return 0
    written = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        snapshot = snapshots.get(entry.get("index"))
        if snapshot is None:
            continue
        original, flags = snapshot
        corrected = str(entry.get("corrected") or "")
        action = entry.get("action") or "changed"
        if not corrected:
            continue
        key = _flags_key(flags)
        if action == "changed" and corrected != original:
            _append({"kind": "correction", "key": key, "original": original,
                     "corrected": corrected, "flags": list(flags),
                     "digits": _digits(original), "shape": _shape(original),
                     "engine": engine, "lang": lang,
                     "created": _now()}, path=path)
            for digits, display, replacement in _value_pairs(original,
                                                             corrected):
                _append({"kind": "value", "value": digits,
                         "display": display, "corrected": replacement,
                         "created": _now()}, path=path)
            written += 1
        suggested = entry.get("suggested")
        if suggested:
            accepted = str(suggested) == corrected
            _append({"kind": "accept" if accepted else "reject", "key": key,
                     "digits": _digits(original),
                     "corrected": str(suggested), "created": _now()},
                    path=path)
            for digits, _display, replacement in _value_pairs(
                    original, str(suggested)):
                _append({"kind": ("value_accept" if accepted
                                  else "value_reject"),
                         "value": digits, "corrected": replacement,
                         "created": _now()}, path=path)
    _trim(path)
    return written


def clear(path: Optional[str] = None) -> int:
    """Delete the local memory; returns the number of events removed."""
    p = path or memory_path()
    removed = len(_iter_events(p))
    try:
        os.remove(p)
    except OSError:
        return 0
    return removed


def stats(path: Optional[str] = None) -> Dict:
    events = _iter_events(path)
    return {
        "enabled": memory_enabled(),
        "events": len(events),
        "corrections": sum(1 for e in events
                           if e.get("kind") == "correction"),
        "values": sum(1 for e in events if e.get("kind") == "value"),
    }
