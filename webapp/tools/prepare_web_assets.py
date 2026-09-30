"""Prepare web assets for the VeriScript studio frontend.

- Downloads the latin woff2 subsets of Instrument Serif + IBM Plex Mono from
  Google Fonts (both OFL) into frontend/public/fonts/.
- Copies Mukta (already bundled by the repo for its PDFs).
- Produces web-sized JPEGs of the real pipeline examples into
  frontend/public/examples/.

    python webapp/tools/prepare_web_assets.py
"""
from __future__ import annotations

import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.dirname(HERE)
REPO = os.path.dirname(WEBAPP)
FONTS_DIR = os.path.join(WEBAPP, "frontend", "public", "fonts")
EX_DIR = os.path.join(WEBAPP, "frontend", "public", "examples")
SRC = os.path.join(REPO, "brag-output", "assets")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_latin_faces(css_url: str) -> list[tuple[dict[str, str], str]]:
    """Google's css2 endpoint returns one @font-face per (style, subset).
    Return [(descriptor dict, woff2 url)] for latin blocks only. The caller
    picks by descriptor — order is not guaranteed."""
    css = fetch(css_url).decode("utf-8")
    blocks = re.findall(r"@font-face\s*{([^}]+)}", css)
    faces: list[tuple[dict[str, str], str]] = []
    for block in blocks:
        ur = re.search(r"unicode-range:\s*([^;]+);", block)
        url = re.search(r"url\((https://[^)]+\.woff2)\)", block)
        if not url:
            continue
        ranges = (ur.group(1) if ur else "").lower()
        if "u+0000-00ff" not in ranges:
            continue
        desc: dict[str, str] = {}
        style = re.search(r"font-style:\s*([^;]+);", block)
        weight = re.search(r"font-weight:\s*([^;]+);", block)
        desc["style"] = (style.group(1).strip() if style else "normal").lower()
        desc["weight"] = (weight.group(1).strip() if weight else "400")
        faces.append((desc, url.group(1)))
    return faces


def pick_face(faces: list[tuple[dict[str, str], str]], *, style: str, weight: str) -> str:
    for desc, url in faces:
        if desc["style"] == style and desc["weight"] == weight:
            return url
    raise SystemExit(f"no latin face for style={style} weight={weight}")


def download(url: str, name: str) -> None:
    dest = os.path.join(FONTS_DIR, name)
    with open(dest, "wb") as f:
        f.write(fetch(url))
    print(f"font  -> {name} ({os.path.getsize(dest)//1024} KB)")


def build_fonts() -> None:
    os.makedirs(FONTS_DIR, exist_ok=True)
    serif = fetch_latin_faces(
        "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap")
    download(pick_face(serif, style="normal", weight="400"), "instrument-serif-400.woff2")
    download(pick_face(serif, style="italic", weight="400"), "instrument-serif-400-italic.woff2")
    mono = fetch_latin_faces(
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap")
    download(pick_face(mono, style="normal", weight="400"), "ibm-plex-mono-400.woff2")
    download(pick_face(mono, style="normal", weight="500"), "ibm-plex-mono-500.woff2")
    mukta_src = os.path.join(REPO, "fonts", "Mukta-Regular.ttf")
    if os.path.isfile(mukta_src):
        import shutil
        shutil.copyfile(mukta_src, os.path.join(FONTS_DIR, "mukta-400.ttf"))
        print("font  -> mukta-400.ttf (bundled by the repo)")


def build_examples() -> None:
    from PIL import Image

    os.makedirs(EX_DIR, exist_ok=True)
    jobs = [
        # (source, out, max_width, quality)
        ("letterpress_input.png", "letterpress-input.jpg", 1500, 84),
        ("letterpress_overlay.png", "letterpress-overlay.jpg", 1500, 84),
        ("invoice_input.png", "invoice-input.jpg", 1200, 84),
        ("invoice_restored.png", "invoice-restored.jpg", 1200, 84),
        ("invoice_overlay.png", "invoice-overlay.jpg", 1200, 84),
        ("letterpress_input.png", "letterpress-hero.jpg", 1100, 82),
        ("invoice_restored.png", "invoice-hero.jpg", 900, 82),
    ]
    clean = [
        # clean English invoice pair lives outside brag-output/assets
        (os.path.join(REPO, "web_outputs", "examples", "synthetic_invoice.png"),
         "synthetic-invoice.jpg", 1240, 86),
        (os.path.join(REPO, "web_outputs", "doc_1790443573649_overlay.png"),
         "synthetic-invoice-overlay.jpg", 1240, 86),
    ]
    for src, out, max_w, q in jobs:
        path = os.path.join(SRC, src)
        if not os.path.isfile(path):
            print(f"skip  -> {src} (missing)")
            continue
        im = Image.open(path).convert("RGB")
        if im.width > max_w:
            h = round(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.LANCZOS)
        dest = os.path.join(EX_DIR, out)
        im.save(dest, "JPEG", quality=q, optimize=True, progressive=True)
        print(f"image -> {out} ({im.width}x{im.height}, {os.path.getsize(dest)//1024} KB)")

    for path, out, max_w, q in clean:
        if not os.path.isfile(path):
            print(f"skip  -> {out} (missing: {path})")
            continue
        im = Image.open(path).convert("RGB")
        if im.width > max_w:
            h = round(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.LANCZOS)
        dest = os.path.join(EX_DIR, out)
        im.save(dest, "JPEG", quality=q, optimize=True, progressive=True)
        print(f"image -> {out} ({im.width}x{im.height}, {os.path.getsize(dest)//1024} KB)")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "fonts"):
        build_fonts()
    if which in ("all", "images"):
        build_examples()


if __name__ == "__main__":
    main()
