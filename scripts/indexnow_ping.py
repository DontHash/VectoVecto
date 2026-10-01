#!/usr/bin/env python3
"""Submit all sitemap URLs to IndexNow (the shared endpoint for Bing, Yandex,
Seznam, Naver — Bing's index is what Copilot and ChatGPT search lean on).

Setup once: put INDEXNOW_KEY=<key> in .env (8-128 hex characters; Bing
Webmaster Tools -> IndexNow can generate one, or any UUID works). The key is
public by design: it is served at https://veriscript.live/<key>.txt, and this
script keeps that file in the frontend's public/ directory.

Run after a deploy:  python scripts/indexnow_ping.py
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITEMAP = ROOT / "webapp" / "frontend" / "public" / "sitemap.xml"
PUBLIC_DIR = ROOT / "webapp" / "frontend" / "public"
ENDPOINT = "https://api.indexnow.org/indexnow"


def load_key() -> str:
    key = os.environ.get("INDEXNOW_KEY", "")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("INDEXNOW_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def sitemap_urls() -> list[str]:
    text = SITEMAP.read_text(encoding="utf-8")
    return re.findall(r"<loc>([^<]+)</loc>", text)


def main() -> None:
    key = load_key()
    if not key:
        raise SystemExit("Missing INDEXNOW_KEY: add it to .env (see .env.example).")

    urls = sitemap_urls()
    if not urls:
        raise SystemExit(f"No <loc> entries found in {SITEMAP.relative_to(ROOT)}.")

    key_path = PUBLIC_DIR / f"{key}.txt"
    key_path.write_text(key, encoding="utf-8")
    print(f"key file: {key_path.relative_to(ROOT)} (include in the build)")

    host = urls[0].split("/")[2]
    key_location = f"https://{host}/{key}.txt"

    try:
        with urllib.request.urlopen(key_location, timeout=15,
                                    context=ssl.create_default_context()) as resp:
            live = resp.status == 200 and key in resp.read(200).decode("utf-8", "replace")
    except Exception:
        live = False
    if not live:
        print(f"warning: {key_location} is not live yet — build + deploy first, "
              "then run this again.")

    body = {"host": host, "key": key, "keyLocation": key_location, "urlList": urls}
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30,
                                    context=ssl.create_default_context()) as resp:
            print(f"IndexNow: HTTP {resp.status} accepted {len(urls)} URLs")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"IndexNow HTTP {exc.code}: {detail[:400]}")


if __name__ == "__main__":
    main()
