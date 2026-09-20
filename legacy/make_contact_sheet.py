"""
make_contact_sheet.py — build comparison images showing all upscale variants
side by side for each test image, plus one combined grid of everything.

Usage:
    python make_contact_sheet.py
"""

# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testImg")

PHOTOS = ["foo1", "foo2", "foo3"]
CONFIGS = [
    ("Original",       "{p}.jpg",                     None),       # no suffix = original
    ("UltraSharp",     "{p}_ultrasharp.png",           None),
    ("RealESRGAN+TTA", "{p}_realesrgan_tta.png",       None),
    ("+Enhance",       "{p}_realesrgan_tta_enhance.png", None),
]

PANEL_HEIGHT = 350          # display height per panel in pixels
LABEL_HEIGHT = 30           # label strip above each panel
GAP = 8                     # gap between panels
BG_COLOR = (30, 30, 30)    # dark background
LABEL_COLOR = (255, 255, 255)
LABEL_BG = (60, 60, 60)


def _load(path):
    """Load an image as RGB PIL."""
    img = Image.open(path).convert("RGB")
    return img


def _resize_to_height(img, target_h):
    """Resize so height = target_h, preserving aspect ratio."""
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def _add_label(img, text):
    """Stick a label strip above the image."""
    label = Image.new("RGB", (img.width, LABEL_HEIGHT), LABEL_BG)
    draw = ImageDraw.Draw(label)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((label.width - tw) // 2, (LABEL_HEIGHT - th) // 2 - 2), text,
              fill=LABEL_COLOR, font=font)
    combined = Image.new("RGB", (img.width, img.height + LABEL_HEIGHT), BG_COLOR)
    combined.paste(label, (0, 0))
    combined.paste(img, (0, LABEL_HEIGHT))
    return combined


def make_per_photo_sheet(photo_name):
    """Build a horizontal 4-panel contact sheet for one photo."""
    panels = []
    for label, path_tmpl, _ in CONFIGS:
        path = os.path.join(TEST_DIR, path_tmpl.format(p=photo_name))
        if not os.path.exists(path):
            print(f"  SKIP {path} (not found)")
            continue
        img = _load(path)
        img = _resize_to_height(img, PANEL_HEIGHT)
        img = _add_label(img, label)
        panels.append(img)

    if not panels:
        return None

    total_w = sum(p.width for p in panels) + GAP * (len(panels) + 1)
    total_h = max(p.height for p in panels) + GAP * 2
    sheet = Image.new("RGB", (total_w, total_h), BG_COLOR)
    x = GAP
    for p in panels:
        sheet.paste(p, (x, GAP))
        x += p.width + GAP

    out_path = os.path.join(TEST_DIR, f"{photo_name}_compare.png")
    sheet.save(out_path, quality=95)
    print(f"  Saved: {out_path} ({total_w}x{total_h})")
    return out_path


def make_combined_grid():
    """Build a 3-row x 4-column grid contact sheet of ALL photos and configs."""
    rows = []
    row_labels = []

    for photo in PHOTOS:
        panels = []
        for label, path_tmpl, _ in CONFIGS:
            path = os.path.join(TEST_DIR, path_tmpl.format(p=photo))
            if not os.path.exists(path):
                placeholder = Image.new("RGB", (400, PANEL_HEIGHT), (50, 50, 50))
                panels.append(_add_label(placeholder, f"{label} (missing)"))
                continue
            img = _load(path)
            img = _resize_to_height(img, PANEL_HEIGHT)
            panels.append(_add_label(img, label))
        rows.append(panels)
        row_labels.append(photo)

    if not rows:
        return None

    # Normalize panel widths to the widest in each column
    n_cols = len(CONFIGS)
    col_widths = []
    for c in range(n_cols):
        cw = max(row[c].width for row in rows if c < len(row))
        col_widths.append(cw)

    total_w = sum(col_widths) + GAP * (n_cols + 1)
    panel_h = max(p.height for row in rows for p in row)
    row_label_w = 80
    total_h = (panel_h + GAP) * len(rows) + GAP

    grid = Image.new("RGB", (total_w + row_label_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for ri, row in enumerate(rows):
        y = GAP + ri * (panel_h + GAP)
        # Row label
        bbox = draw.textbbox((0, 0), row_labels[ri], font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((10, y + panel_h // 2 - th // 2), row_labels[ri],
                  fill=(255, 255, 100), font=font)

        x = row_label_w + GAP
        for ci, panel in enumerate(row):
            # Center the panel in its column slot
            px = x + (col_widths[ci] - panel.width) // 2
            grid.paste(panel, (px, y))
            x += col_widths[ci] + GAP

    out_path = os.path.join(TEST_DIR, "ALL_compare.png")
    grid.save(out_path, quality=95)
    print(f"\nCombined grid saved: {out_path} ({grid.width}x{grid.height})")
    return out_path


if __name__ == "__main__":
    print("=== Building per-photo contact sheets ===")
    for photo in PHOTOS:
        print(f"\n[{photo}]")
        make_per_photo_sheet(photo)

    print("\n=== Building combined grid ===")
    make_combined_grid()
    print("\nDone.")