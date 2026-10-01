"""Frontend shell guards: the crawler fallback must never depend on inline code.

Regression (2026-10-01): index.html removed #seo-fallback with an inline
script, but the production CSP is `script-src 'self'` — the browser blocked the
script, the fallback stayed in the DOM below Solid's appended app and rendered
above it. The fallback must be removed by the bundled module (main.tsx) and
hidden by the bundled stylesheet; index.html must stay inline-script-free.
"""
from __future__ import annotations

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE_DIR, "webapp", "frontend")


def _read(*parts: str) -> str:
    with open(os.path.join(FRONTEND, *parts), encoding="utf-8") as f:
        return f.read()


_DATA_BLOCK = re.compile(
    r"""type\s*=\s*["'](application/ld\+json|application/json|text/template)["']""")


def test_index_html_has_no_inline_scripts():
    html = _read("index.html")
    offenders = []
    for attrs in re.findall(r"<script\b([^>]*)>", html):
        if "src=" in attrs or _DATA_BLOCK.search(attrs):
            continue  # external module or a non-executable data block (JSON-LD)
        offenders.append(attrs.strip())
    assert not offenders, \
        f"inline executable scripts are blocked by the CSP: {offenders}"


def test_crawler_fallback_present_and_removed_by_the_bundle():
    assert 'id="seo-fallback"' in _read("index.html"), \
        "crawlers need the static fallback in the raw HTML"
    main = _read("src", "main.tsx")
    assert "seo-fallback" in main and ".remove()" in main, \
        "the app bundle must remove the fallback before mount"


def test_fallback_is_hidden_by_the_bundled_stylesheet():
    assert "#seo-fallback" in _read("src", "styles", "base.css")
