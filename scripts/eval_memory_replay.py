#!/usr/bin/env python3
"""eval_memory_replay.py — offline replay of the local correction memory
(track D of docs/HARNESS_PLAN.md) over the Appendix AI beta sessions.

Memory accumulates from earlier pages; each page's queue is queried before it
is fed back. Three channels are measured:

  line      exact reading / same-skeleton matching (whole-token suggestion)
  value     a repeated wrong digit run substituted in place
  combined  what `augment_review` actually offers (line first, then value)

A "full hit" means the suggestion equals the human's actual outcome (the
corrected text, or the original when the token was confirmed). A "run hit"
means the suggestion's digit runs equal the human's digit runs - the value
fix is right even if other words still differ (value channel only).

This is product evidence, not a frozen-set gate: the beta pages are
non-frozen by construction (`scripts/corrections_beta.py` refuses frozen ids).

Usage:
    python scripts/eval_memory_replay.py --sessions-dir out/beta
    python scripts/eval_memory_replay.py --sessions-dir out/beta --json out/memory_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document import memory  # noqa: E402
from veriscript.lexicon import deva_words  # noqa: E402

_RUN_RE = re.compile(
    r"[0-9\u0966-\u096f](?:[0-9\u0966-\u096f.,:/-]*[0-9\u0966-\u096f])?")


def _runs(text: str) -> List[str]:
    return [m.group(0) for m in _RUN_RE.finditer(text or "")]


def replay(sessions_dir: str, memory_file: Optional[str] = None) -> Dict:
    if memory_file is None:
        fd, memory_file = tempfile.mkstemp(prefix="veriscript_memory_",
                                           suffix=".jsonl")
        os.close(fd)
        os.remove(memory_file)
    os.environ["VERISCRIPT_MEMORY"] = "1"
    os.environ["VERISCRIPT_MEMORY_DIR"] = memory_file

    sessions = sorted(d for d in os.listdir(sessions_dir)
                      if d.startswith("s")
                      and os.path.isdir(os.path.join(sessions_dir, d)))
    rows: List[Dict] = []
    totals = {"queued": 0,
              "line_suggested": 0, "line_hits": 0,
              "value_suggested": 0, "value_hits": 0, "value_run_hits": 0,
              "word_suggested": 0, "word_hits": 0, "word_word_hits": 0,
              "combined_suggested": 0, "combined_hits": 0}
    for session in sessions:
        queue = json.load(open(os.path.join(sessions_dir, session, "queue.json"),
                               encoding="utf-8"))
        corr = json.load(open(os.path.join(sessions_dir, session,
                                           "beta_corrections.json"),
                              encoding="utf-8"))
        applied = {e["index"]: e for e in corr.get("applied", [])}
        counts = dict.fromkeys(totals, 0)
        for item in queue:
            totals["queued"] += 1
            counts["queued"] += 1
            text = item.get("text", "")
            flags = item.get("flags") or []
            line = memory.suggest(text, flags, path=memory_file)
            value = memory.suggest_value(text, path=memory_file)
            word = memory.suggest_word(text, path=memory_file)
            combined = line or value or word
            entry = applied.get(item["index"])
            target = None
            if entry is not None:
                target = (entry.get("corrected", "")
                          if entry.get("action") == "changed"
                          else entry.get("original", text))

            for channel, suggestion in (("line", line), ("value", value),
                                        ("word", word),
                                        ("combined", combined)):
                if suggestion is None:
                    continue
                totals[f"{channel}_suggested"] += 1
                counts[f"{channel}_suggested"] += 1
                if target is not None and \
                        suggestion["text"].strip() == (target or "").strip():
                    totals[f"{channel}_hits"] += 1
                    counts[f"{channel}_hits"] += 1
            if value is not None and target is not None:
                if _runs(value["text"]) == _runs(target):
                    totals["value_run_hits"] += 1
                    counts["value_run_hits"] += 1
            if word is not None and target is not None:
                if word.get("word") in set(deva_words(target)):
                    totals["word_word_hits"] += 1
                    counts["word_word_hits"] += 1
        rows.append({"session": session, **counts})

        entries = [{"index": e["index"], "corrected": e.get("corrected", ""),
                    "action": e.get("action", "changed")}
                   for e in corr.get("applied", [])]
        snaps = {item["index"]: (item.get("text", ""),
                                 item.get("flags") or [])
                 for item in queue}
        memory.record_corrections(entries, snaps, engine="rapidocr",
                                  lang="ne", path=memory_file)

    def _rate(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    summary = {
        "sessions": len(rows),
        "queued": totals["queued"],
        "line": {"suggested": totals["line_suggested"],
                 "hits": totals["line_hits"],
                 "coverage": _rate(totals["line_suggested"],
                                   totals["queued"]),
                 "precision": _rate(totals["line_hits"],
                                    totals["line_suggested"])},
        "value": {"suggested": totals["value_suggested"],
                  "hits": totals["value_hits"],
                  "run_hits": totals["value_run_hits"],
                  "coverage": _rate(totals["value_suggested"],
                                    totals["queued"]),
                  "precision": _rate(totals["value_hits"],
                                     totals["value_suggested"]),
                  "run_precision": _rate(totals["value_run_hits"],
                                         totals["value_suggested"])},
        "word": {"suggested": totals["word_suggested"],
                 "hits": totals["word_hits"],
                 "word_hits": totals["word_word_hits"],
                 "coverage": _rate(totals["word_suggested"],
                                   totals["queued"]),
                 "precision": _rate(totals["word_hits"],
                                    totals["word_suggested"]),
                 "word_precision": _rate(totals["word_word_hits"],
                                         totals["word_suggested"])},
        "combined": {"suggested": totals["combined_suggested"],
                     "hits": totals["combined_hits"],
                     "coverage": _rate(totals["combined_suggested"],
                                       totals["queued"]),
                     "precision": _rate(totals["combined_hits"],
                                        totals["combined_suggested"])},
    }
    return {"summary": summary, "sessions": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="offline correction-memory replay")
    ap.add_argument("--sessions-dir", default="out/beta")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    result = replay(args.sessions_dir)
    for row in result["sessions"]:
        print(f"  {row['session']}: queued {row['queued']}, "
              f"line {row['line_suggested']}/{row['line_hits']}, "
              f"value {row['value_suggested']}/{row['value_hits']} "
              f"(runs {row['value_run_hits']}), "
              f"word {row['word_suggested']}/{row['word_word_hits']}")
    s = result["summary"]
    print(f"TOTAL queued {s['queued']}")
    for channel in ("line", "value", "word", "combined"):
        c = s[channel]
        extras = []
        if c.get("run_precision") is not None:
            extras.append(f"run-precision {c['run_precision']:.1%}")
        if c.get("word_precision") is not None:
            extras.append(f"word-precision {c['word_precision']:.1%}")
        extra = (" " + " ".join(extras)) if extras else ""
        print(f"  {channel:<8}: suggested {c['suggested']} "
              f"({c['coverage']:.1%}), full hits {c['hits']} "
              f"(precision {c['precision']:.1%}){extra}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[memory-replay] wrote {args.json}")


if __name__ == "__main__":
    main()
