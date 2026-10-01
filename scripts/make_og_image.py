#!/usr/bin/env python3
"""Generate webapp/frontend/public/og.png — the 1200x630 social card.

Same palette as the studio (warm paper, one ink, one amber accent; the
green/amber/red trio is data color and only appears in the review legend).
Regenerate after branding changes:  python scripts/make_og_image.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "webapp" / "frontend" / "public" / "og.png"

W, H = 1200, 630
PAPER = (244, 240, 230)
INK = (18, 14, 8)
INK_2 = (68, 62, 51)
INK_3 = (84, 79, 69)
AMBER = (232, 162, 0)
GREEN = (0, 138, 0)
RED = (212, 0, 0)

GEORGIA = ["C:/Windows/Fonts/georgiab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]
MUKTA = [str(ROOT / "webapp" / "frontend" / "public" / "fonts" / "mukta-400.ttf")]
MONO = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 10], fill=AMBER)

    kicker = font(MONO, 22)
    title = font(GEORGIA, 62)
    body = font(MUKTA, 28)
    small = font(MONO, 20)

    x = 80
    draw.text((x, 92), "VERISCRIPT · OFFLINE DOCUMENT RESTORATION", font=kicker, fill=INK_3)

    draw.text((x, 150), "It will not invent", font=title, fill=INK)
    draw.text((x, 222), "the numbers on your bill.", font=title, fill=INK)

    draw.text((x, 330), "Nepali / Hindi OCR with a review queue", font=body, fill=INK_2)
    draw.text((x, 370), "for uncertain digits. Searchable PDF out —", font=body, fill=INK_2)
    draw.text((x, 410), "offline, no account, MIT.", font=body, fill=INK_2)

    # review-queue legend (the only place data colors appear)
    y = 500
    for sx, color, label in [
        (x, GREEN, "accepted"),
        (x + 220, AMBER, "uncertain"),
        (x + 470, RED, "digit conflict"),
    ]:
        draw.rectangle([sx, y + 4, sx + 18, y + 22], fill=color)
        draw.text((sx + 30, y), label, font=small, fill=INK_3)

    footer = "veriscript.live · github.com/DontHash/VeriScript"
    width = draw.textlength(footer, font=small)
    draw.text((W - 80 - width, 560), footer, font=small, fill=INK_3)

    img.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
