"""
build_phase4_5_comparison.py — Generate high-resolution side-by-side comparison sheets
comparing Original, Bicubic, Tier-B Deep Unfolding, and Phase 4.5 Hybrid Vector/Raster.
"""


# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def add_label(img_pil, text, font_size=16, bg_color=(45, 45, 45), text_color=(255, 255, 255)):
    label_h = 28
    label = Image.new("RGB", (img_pil.width, label_h), bg_color)
    draw = ImageDraw.Draw(label)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((img_pil.width - tw) // 2, (label_h - th) // 2 - 1), text, fill=text_color, font=font)
    
    combined = Image.new("RGB", (img_pil.width, img_pil.height + label_h), (30, 30, 30))
    combined.paste(label, (0, 0))
    combined.paste(img_pil, (0, label_h))
    return combined

def build_de1_comparison():
    print("Building De1 comparison sheet (Before vs After Clay Fix)...")
    lr = cv2.imread("De1.jpg")
    h, w = lr.shape[:2]
    out_h, out_w = 628, 940

    # 1. Bicubic Baseline
    bic = cv2.resize(lr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # 2. Old Legacy (Clay / Bilateral / TV)
    clay_path = "De1_drunet_tv.png" if os.path.exists("De1_drunet_tv.png") else "De1_drunet_upscaled.png"
    clay_img = cv2.imread(clay_path)
    if clay_img is not None and clay_img.shape[:2] != (out_h, out_w):
        clay_img = cv2.resize(clay_img, (out_w, out_h))

    # 3. New Natural DRUNet (Organic Grain, No Bilateral Filter)
    nat_path = "De1_natural_drunet.png" if os.path.exists("De1_natural_drunet.png") else "De1_raw_drunet.png"
    nat_img = cv2.imread(nat_path)
    if nat_img is not None and nat_img.shape[:2] != (out_h, out_w):
        nat_img = cv2.resize(nat_img, (out_w, out_h))
    elif nat_img is None:
        nat_img = bic

    # 4. Phase 4.5 Hybrid (Natural Raster + Bézier Vector Typography)
    hybrid = cv2.imread("De1_clean_hybrid.png") if os.path.exists("De1_clean_hybrid.png") else cv2.imread("De1_hybrid_4x.png")

    # Crop coordinates around "JAMES 23" on jersey: y: 260..580, x: 500..840
    cy1, cy2, cx1, cx2 = 260, 580, 500, 840

    crop_bic = bic[cy1:cy2, cx1:cx2]
    crop_clay = clay_img[cy1:cy2, cx1:cx2] if clay_img is not None else crop_bic
    crop_nat = nat_img[cy1:cy2, cx1:cx2]
    crop_hyb = hybrid[cy1:cy2, cx1:cx2]

    # Convert to PIL
    def to_pil(arr):
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    panel_w = 320
    panel_h = int(out_h * (panel_w / float(out_w)))

    p_lr = to_pil(cv2.resize(lr, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST))
    p_bic = to_pil(cv2.resize(bic, (panel_w, panel_h), interpolation=cv2.INTER_AREA))
    p_clay = to_pil(cv2.resize(clay_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)) if clay_img is not None else p_bic
    p_nat = to_pil(cv2.resize(nat_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA))
    p_hyb = to_pil(cv2.resize(hybrid, (panel_w, panel_h), interpolation=cv2.INTER_AREA))

    # Zoomed crops (enlarged for crisp view)
    zoom_w = 340
    zoom_h = int((cy2 - cy1) * (zoom_w / float(cx2 - cx1)))

    z_bic = to_pil(cv2.resize(crop_bic, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST))
    z_clay = to_pil(cv2.resize(crop_clay, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST))
    z_nat = to_pil(cv2.resize(crop_nat, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST))
    z_hyb = to_pil(cv2.resize(crop_hyb, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST))

    # Build comparison grid
    # Row 1: Full images (5 columns)
    row1_items = [
        add_label(p_lr, "Original LR (235x157)", bg_color=(50, 50, 50)),
        add_label(p_bic, "Bicubic 4x Baseline", bg_color=(50, 50, 50)),
        add_label(p_clay, "Old Legacy (Clay / Plastic)", bg_color=(80, 30, 30)),
        add_label(p_nat, "New Natural DRUNet", bg_color=(20, 70, 50)),
        add_label(p_hyb, "Phase 4.5 Hybrid Vector/Raster", bg_color=(70, 40, 20)),
    ]

    # Row 2: Zoomed Crops (4 columns)
    row2_items = [
        add_label(z_bic, "Bicubic Zoom (Blurry)", bg_color=(50, 50, 50)),
        add_label(z_clay, "Old Legacy Zoom (Clay Plates)", bg_color=(80, 30, 30)),
        add_label(z_nat, "Natural Zoom (Photographic Skin)", bg_color=(20, 70, 50)),
        add_label(z_hyb, "Phase 4.5 Zoom (Crisp Bézier)", bg_color=(70, 40, 20)),
    ]

    gap = 10
    margin = 16

    row1_w = sum(it.width for it in row1_items) + gap * (len(row1_items) - 1)
    row1_h = max(it.height for it in row1_items)

    row2_w = sum(it.width for it in row2_items) + gap * (len(row2_items) - 1)
    row2_h = max(it.height for it in row2_items)

    canvas_w = max(row1_w, row2_w) + margin * 2
    header_h = 75
    canvas_h = header_h + row1_h + gap * 2 + row2_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        subtitle_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    draw.text((margin, 12), "VectorScaling — Eliminating the 'Clay-ish' / Plastic Artifact", fill=(255, 255, 255), font=title_font)
    draw.text((margin, 42), "Root Cause Fix: Bypassed hardcoded bilateral filter & unsharp haloing | Injected organic film micro-grain", fill=(180, 180, 180), font=subtitle_font)

    # Paste Row 1
    x_cur = margin + (canvas_w - margin * 2 - row1_w) // 2
    y_cur = header_h
    for it in row1_items:
        canvas.paste(it, (x_cur, y_cur))
        x_cur += it.width + gap

    # Paste Row 2
    x_cur = margin + (canvas_w - margin * 2 - row2_w) // 2
    y_cur = header_h + row1_h + gap * 2
    for it in row2_items:
        canvas.paste(it, (x_cur, y_cur))
        x_cur += it.width + gap

    out_file = "De1_detailed_comparison.png"
    canvas.save(out_file, quality=95)
    print(f"  Saved: {out_file} ({canvas.width}x{canvas.height})")
    return out_file

def build_synthetic_comparison():
    print("Building Synthetic Benchmark comparison sheet...")
    if not (os.path.exists("test_hr.png") and os.path.exists("test_bicubic.png") and os.path.exists("test_hybrid_4_5.png")):
        print("  Skipping synthetic: files not found")
        return None

    hr = cv2.imread("test_hr.png")
    lr = cv2.imread("test_lr.png")
    bic = cv2.imread("test_bicubic.png")
    hyb = cv2.imread("test_hybrid_4_5.png")

    def to_pil(arr):
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    panel_size = 360
    p_hr = to_pil(cv2.resize(hr, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST))
    p_lr = to_pil(cv2.resize(lr, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST))
    p_bic = to_pil(cv2.resize(bic, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST))
    p_hyb = to_pil(cv2.resize(hyb, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST))

    items = [
        add_label(p_hr, "Ground Truth HR", bg_color=(20, 70, 50)),
        add_label(p_lr, "Low-Res Input (Degraded)", bg_color=(50, 50, 50)),
        add_label(p_bic, "Bicubic 2x (Blurry Graphics)", bg_color=(70, 30, 30)),
        add_label(p_hyb, "Phase 4.5 Hybrid (Sharp Bézier)", bg_color=(70, 40, 20)),
    ]

    gap = 12
    margin = 16
    header_h = 60

    total_w = sum(it.width for it in items) + gap * (len(items) - 1) + margin * 2
    total_h = max(it.height for it in items) + header_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        subtitle_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    draw.text((margin, 10), "Synthetic Benchmark — Graphic & Texture Multi-Domain Test", fill=(255, 255, 255), font=title_font)
    draw.text((margin, 36), "Evaluates: Text ('VECTOR 4.5'), Nested Badges, Star Polygon + Background Fabric Texture", fill=(180, 180, 180), font=subtitle_font)

    x_cur = margin
    for it in items:
        canvas.paste(it, (x_cur, header_h))
        x_cur += it.width + gap

    out_file = "Synthetic_benchmark_comparison.png"
    canvas.save(out_file, quality=95)
    print(f"  Saved: {out_file} ({canvas.width}x{canvas.height})")
    return out_file

if __name__ == "__main__":
    build_de1_comparison()
    build_synthetic_comparison()
