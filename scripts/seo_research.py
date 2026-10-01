#!/usr/bin/env python3
"""DataForSEO first-pass research for the discoverability plan (docs/SEO_AEO.md).

Reads credentials from .env at the repo root (DATAFORSEO_LOGIN /
DATAFORSEO_PASSWORD) or from the environment. Standard library only.

Modes:
  check     -- balance + one live SERP. Cents. Validates the credentials.
  keywords  -- search volume for the seed set (one live request).
  serps     -- top-10 SERPs for the priority queries (live).
  all       -- keywords + serps; writes data/seo_research/ (JSON + summary.md).

Cost notes: a live Google Ads search-volume request is charged per request
(1 or 1000 keywords cost the same); a live SERP request is ~$0.002. A full
`all` pass is well under $0.25 at current rates.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "seo_research"
API = "https://api.dataforseo.com/v3"

# ---------------------------------------------------------------- seed sets

# One Google Ads request holds up to 1000 keywords; keep the phrases <= 10
# words and avoid mixed similar phrases where separation matters later.
SEED_KEYWORDS = [
    # core script / product queries
    "nepali ocr",
    "nepali ocr online",
    "free nepali ocr",
    "best nepali ocr",
    "devanagari ocr",
    "hindi ocr",
    "nepali image to text",
    "photo to text nepali",
    "nepali pdf to text",
    "nepali pdf to word",
    "searchable pdf nepali",
    "make scanned pdf searchable",
    "nepali document scan to text",
    "nepali document scanner",
    # use cases
    "citizenship card ocr",
    "lalpurja ocr",
    "nepali newspaper archive",
    "digitize nepali documents",
    "nepali handwriting to text",
    "handwriting to text nepali",
    # privacy / offline
    "offline ocr software",
    "ocr without internet",
    "private ocr",
    "self hosted ocr",
    "air gapped ocr",
    "ocr no cloud",
    "gdpr compliant ocr",
    # honesty / accuracy
    "ocr that detects errors",
    "ocr digit verification",
    "ocr error detection",
    "ocr verification tool",
    "ocr accuracy check",
    # developers
    "tesseract nepali",
    "paddleocr devanagari",
    "devanagari ocr python",
    "ocr python library",
    "pdf to markdown",
    "pdf to markdown open source",
    "searchable pdf python",
    "ocr json output",
    # alternatives
    "abbyy finereader alternative",
    "tesseract alternative",
    "ocrmypdf alternative",
    "google vision ocr alternative",
    "acrobat ocr alternative",
    "free ocr software offline",
    # AI-era
    "chatgpt ocr",
    "ai document reader offline",
    "llm ocr",
]

# Priority queries for live SERP inspection: (query, location).
# Nepal first for all; the diaspora markets matter for the top queries.
SERP_PLAN = [
    ("nepali ocr", "Nepal"),
    ("nepali ocr", "United States"),
    ("nepali image to text", "Nepal"),
    ("nepali image to text", "United States"),
    ("devanagari ocr", "Nepal"),
    ("devanagari ocr", "United States"),
    ("hindi ocr", "Nepal"),
    ("hindi ocr", "United States"),
    ("offline ocr software", "Nepal"),
    ("offline ocr software", "United States"),
    ("private ocr", "Nepal"),
    ("searchable pdf nepali", "Nepal"),
    ("pdf to markdown", "Nepal"),
    ("abbyy finereader alternative", "Nepal"),
    ("nepali handwriting to text", "Nepal"),
]

# ---------------------------------------------------------------- plumbing


def load_env() -> tuple[str, str]:
    login = os.environ.get("DATAFORSEO_LOGIN", "")
    password = os.environ.get("DATAFORSEO_PASSWORD", "")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key.strip() == "DATAFORSEO_LOGIN" and not login:
                login = value
            elif key.strip() == "DATAFORSEO_PASSWORD" and not password:
                password = value
    return login, password


def api_call(auth: tuple[str, str], path: str, payload=None) -> dict:
    url = f"{API}/{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=90,
                                    context=ssl.create_default_context()) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}\n{detail[:2000]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error calling {url}: {exc.reason}")
    parsed = json.loads(body)
    if parsed.get("status_code") != 20000:
        raise SystemExit(
            f"DataForSEO error {parsed.get('status_code')}: {parsed.get('status_message')}"
        )
    return parsed


def task_of(response: dict) -> dict:
    task = response["tasks"][0]
    if task.get("status_code") != 20000:
        raise SystemExit(
            f"Task error {task.get('status_code')}: {task.get('status_message')}"
        )
    return task


def user_data(auth) -> dict:
    task = task_of(api_call(auth, "appendix/user_data"))
    result = task["result"][0]
    money = result.get("money", {})
    balance = money.get("balance")
    print(f"login:   {result.get('login')}")
    print(f"balance: {balance} {money.get('currency', '')} "
          f"(deposited: {money.get('total', 'n/a')})")
    return result


def keywords_live(auth, keywords: list[str], location: str) -> list[dict]:
    payload = [{
        "keywords": keywords,
        "location_name": location,
        "language_code": "en",
    }]
    task = task_of(api_call(auth, "keywords_data/google_ads/search_volume/live", payload))
    rows = [r for r in (task.get("result") or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: r.get("search_volume") or 0, reverse=True)
    return rows


def serp_live(auth, keyword: str, location: str, depth: int = 10) -> list[dict]:
    payload = [{
        "keyword": keyword,
        "location_name": location,
        "language_code": "en",
        "device": "desktop",
        "depth": depth,
    }]
    task = task_of(api_call(auth, "serp/google/organic/live/advanced", payload))
    result = (task.get("result") or [None])[0]
    if not result:
        return []
    rows = []
    for item in result.get("items") or []:
        if item.get("type") == "organic":
            rows.append({
                "rank": item.get("rank_absolute"),
                "domain": item.get("domain"),
                "url": item.get("url"),
                "title": item.get("title"),
            })
    return rows


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_json(name: str, payload) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

# ---------------------------------------------------------------- modes


def run_check(auth) -> None:
    user_data(auth)
    rows = serp_live(auth, "nepali ocr", "Nepal")
    print("\nSERP 'nepali ocr' (Nepal):")
    for row in rows:
        print(f"  {row['rank']:>2}. {row['domain']}  {row['title'] or ''}")
    print("\nCheck passed. Run 'all' for the full first pass.")


def run_keywords(auth, location: str, lines: list[str]) -> None:
    print(f"Search volume ({location}, {len(SEED_KEYWORDS)} keywords)...")
    rows = keywords_live(auth, SEED_KEYWORDS, location)
    path = save_json(f"keywords_{slug(location)}.json", rows)
    lines.append(f"## Keyword volumes - {location}\n")
    lines.append("| volume | cpc | competition | keyword |")
    lines.append("|---:|---:|---|---|")
    for row in rows:
        cpc = row.get("cpc")
        cpc_text = "-" if cpc is None else f"${cpc:.2f}"
        competition = row.get("competition") or "-"
        volume = row.get("search_volume")
        volume_text = "-" if volume is None else str(volume)
        lines.append(f"| {volume_text} | {cpc_text} | {competition} | {row.get('keyword')} |")
    lines.append(f"\nRaw: `{path.relative_to(ROOT).as_posix()}`\n")
    for row in rows[:15]:
        print(f"  {str(row.get('search_volume')):>8}  {row.get('keyword')}")
    print(f"  ... {len(rows)} rows total -> {path.relative_to(ROOT)}")


def run_serps(auth, lines: list[str]) -> None:
    print(f"SERP pass ({len(SERP_PLAN)} live queries)...")
    for query, location in SERP_PLAN:
        rows = serp_live(auth, query, location)
        path = save_json(f"serp_{slug(query)}_{slug(location)}.json",
                         {"query": query, "location": location, "results": rows})
        lines.append(f"### \"{query}\" - {location}\n")
        if not rows:
            lines.append("_no organic results returned_\n")
        else:
            lines.append("| # | domain | title |")
            lines.append("|---:|---|---|")
            for row in rows:
                title = (row.get("title") or "").replace("|", "/")
                lines.append(f"| {row['rank']} | {row['domain']} | {title} |")
            lines.append("")
        lines.append(f"Raw: `{path.relative_to(ROOT).as_posix()}`\n")
        top = rows[0]["domain"] if rows else "-"
        print(f"  {'ok ' if rows else '!! '} {query!r} [{location}]  #1: {top}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="DataForSEO first-pass research")
    parser.add_argument("mode", choices=["check", "keywords", "serps", "all"])
    parser.add_argument("--location", default="Nepal",
                        help="location for the keyword-volume request")
    args = parser.parse_args()

    login, password = load_env()
    if not login or not password:
        raise SystemExit(
            "Missing credentials: put DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD "
            "into .env at the repo root (see .env.example)."
        )

    auth = (login, password)
    if args.mode == "check":
        run_check(auth)
        return

    lines = [f"# DataForSEO first pass - {date.today().isoformat()}", ""]
    if args.mode in ("keywords", "all"):
        run_keywords(auth, args.location, lines)
    if args.mode in ("serps", "all"):
        run_serps(auth, lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = OUT_DIR / "summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary: {summary.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
