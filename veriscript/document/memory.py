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
  {"kind": "accept"|"reject", "key": <flags signature>,
   "digits": <digits of the original reading>, "corrected": <suggested text>,
   "created": ...}

Suggestion matching: same flags signature, exact digits first, else same
digit length and Hamming distance <= 1. A record whose feedback is
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
    """Append memory suggestions to review rows in place. Returns rows gained."""
    if not memory_enabled() or not review:
        return 0
    index = _index(_iter_events(path))
    added = 0
    for item in review:
        if not isinstance(item, dict):
            continue
        match = _lookup(index, item.get("text") or "", item.get("flags") or [])
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
            written += 1
        suggested = entry.get("suggested")
        if suggested:
            accepted = str(suggested) == corrected
            _append({"kind": "accept" if accepted else "reject", "key": key,
                     "digits": _digits(original),
                     "corrected": str(suggested), "created": _now()},
                    path=path)
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
    }
