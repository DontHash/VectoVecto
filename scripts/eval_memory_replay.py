#!/usr/bin/env python3
"""eval_memory_replay.py — offline replay of the local correction memory
(track D2 of docs/HARNESS_PLAN.md) over the Appendix AI beta sessions.

Memory accumulates from earlier pages; each page's queue is queried before it
is fed back. A "hit" means the suggestion equals the human's actual outcome
(the corrected text, or the original when the token was confirmed). This is
product evidence, not a frozen-set gate: the beta pages are non-frozen by
construction (`scripts/corrections_beta.py` refuses frozen page ids).

Usage:
    python scripts/eval_memory_replay.py --sessions-dir out/beta
    python scripts/eval_memory_replay.py --sessions-dir out/beta --json out/memory_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from veriscript.document import memory  # noqa: E402


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
    total = covered = hits = 0
    for session in sessions:
        queue = json.load(open(os.path.join(sessions_dir, session, "queue.json"),
                               encoding="utf-8"))
        corr = json.load(open(os.path.join(sessions_dir, session,
                                           "beta_corrections.json"),
                              encoding="utf-8"))
        applied = {e["index"]: e for e in corr.get("applied", [])}
        s_covered = s_hits = 0
        for item in queue:
            total += 1
            suggestion = memory.suggest(item.get("text", ""),
                                        item.get("flags") or [],
                                        path=memory_file)
            if suggestion is None:
                continue
            covered += 1
            s_covered += 1
            entry = applied.get(item["index"])
            if entry is None:
                continue
            target = (entry.get("corrected", "")
                      if entry.get("action") == "changed"
                      else entry.get("original", item.get("text", "")))
            if suggestion["text"].strip() == (target or "").strip():
                hits += 1
                s_hits += 1
        rows.append({"session": session, "queued": len(queue),
                     "suggested": s_covered, "hits": s_hits})

        entries = [{"index": e["index"], "corrected": e.get("corrected", ""),
                    "action": e.get("action", "changed")}
                   for e in corr.get("applied", [])]
        snaps = {item["index"]: (item.get("text", ""),
                                 item.get("flags") or [])
                 for item in queue}
        memory.record_corrections(entries, snaps, engine="rapidocr",
                                  lang="ne", path=memory_file)

    summary = {
        "sessions": len(rows),
        "queued": total,
        "suggested": covered,
        "hits": hits,
        "coverage": round(covered / total, 4) if total else None,
        "precision": round(hits / covered, 4) if covered else None,
        "resolved": round(hits / total, 4) if total else None,
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
              f"suggested {row['suggested']}, hits {row['hits']}")
    summary = result["summary"]
    print(f"TOTAL queued {summary['queued']} | suggested "
          f"{summary['suggested']} ({summary['coverage']:.1%}) | hits "
          f"{summary['hits']} | precision {summary['precision']:.1%} | "
          f"resolved {summary['resolved']:.1%}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[memory-replay] wrote {args.json}")


if __name__ == "__main__":
    main()
