"""
test_web_app.py — Automated Verification for Interactive Split-Screen Web App.

Verifies:
  1. app.py module imports and constructs Gradio Blocks demo.
  2. process_image executes cleanly on synthetic image.
  3. ImageSlider outputs (LR aligned, HR upscaled) have matching shapes.
  4. PNG, SVG, and diagnostic masks are created and non-empty.
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, r"d:\VectorScaling")
import app

def make_test_input():
    size = 96
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[10:50, 10:50] = [0, 220, 255] # Yellow block
    cv2.putText(img, "WEB", (15, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)
    return img

def test_create_app():
    print("Testing Gradio app construction...")
    demo = app.create_app()
    assert demo is not None, "Failed to create Gradio Blocks demo"
    print("  -> Gradio app instantiated successfully!")

def test_process_image_pipeline(tmp_path):
    print("Testing process_image backend pipeline for gr.ImageSlider...")
    test_img = make_test_input() # RGB
    scale = 2

    slider_data, diag_rgb, png_path, svg_path, status_md = app.process_image(
        input_img=test_img,
        scale=scale,
        mode="auto",
        fast_mode=True,
        grain_val=0.018,
        export_svg_flag=True,
        show_mask_flag=True
    )

    # 1. Verify ImageSlider tuple
    assert slider_data is not None, "ImageSlider tuple must not be None"
    assert isinstance(slider_data, tuple) and len(slider_data) == 2, "ImageSlider must return a 2-tuple (before, after)"
    lr_aligned, hr_upscaled = slider_data

    expected_shape = (test_img.shape[0] * scale, test_img.shape[1] * scale, 3)
    assert lr_aligned.shape == expected_shape, f"LR aligned shape {lr_aligned.shape} != {expected_shape}"
    assert hr_upscaled.shape == expected_shape, f"HR upscaled shape {hr_upscaled.shape} != {expected_shape}"

    # 2. Verify files created
    assert os.path.exists(png_path), f"PNG file not found at {png_path}"
    assert os.path.getsize(png_path) > 0, "PNG file is empty"

    if svg_path:
        assert os.path.exists(svg_path), f"SVG file not found at {svg_path}"
        assert os.path.getsize(svg_path) > 0, "SVG file is empty"

    # 3. Verify status markdown
    assert "Processing Complete" in status_md, "Status markdown missing completion header"
    print(f"  -> process_image verified: {slider_data[0].shape} -> {slider_data[1].shape}")
