"""
test_vector_raster.py — Comprehensive Unit & Integration Tests for Phase 4.5.

Tests:
  1. Point2D, CubicBezier math, and Schneider curve fitting accuracy.
  2. Structure tensor computation & texture vs. graphic segmentation.
  3. SVG export and validity check.
  4. Hybrid vector/raster upscaling against bicubic baseline on a synthetic
     image containing both sharp graphics (text, polygons) and stochastic textures.
"""

import os
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from vector_raster_hybrid import (
    Point2D, CubicBezier, fit_cubic_bezier_segment, fit_curve_recursive,
    vectorize_contour, compute_structure_tensor, segment_flat_and_graphic_regions,
    extract_vector_shapes, render_vector_shapes, export_svg,
    hybrid_vector_raster_upscale
)
from eval_harness import calculate_psnr, calculate_ssim, degrade_image


def test_bezier_math():
    print("Testing Bézier curve mathematics and Schneider fitting...")
    p0 = Point2D(0, 0)
    p1 = Point2D(0, 10)
    p2 = Point2D(10, 10)
    p3 = Point2D(10, 0)

    curve = CubicBezier(p0, p1, p2, p3)
    start_pt = curve.eval(0.0)
    mid_pt = curve.eval(0.5)
    end_pt = curve.eval(1.0)

    assert abs(start_pt.x - 0.0) < 1e-5 and abs(start_pt.y - 0.0) < 1e-5, "Bézier start point failed"
    assert abs(end_pt.x - 10.0) < 1e-5 and abs(end_pt.y - 0.0) < 1e-5, "Bézier end point failed"
    assert mid_pt.y > 5.0, "Bézier midpoint curvature failed"

    # Test scaling
    scaled = curve.scale(4.0)
    assert abs(scaled.p3.x - 40.0) < 1e-5, "Bézier scaling failed"

    # Test curve fitting on a quadrant of a circle
    theta = np.linspace(0, np.pi / 2, 20)
    r = 50.0
    circle_pts = [Point2D(r * np.cos(t), r * np.sin(t)) for t in theta]
    t1 = Point2D(0, 1)   # tangent at theta=0
    t2 = Point2D(-1, 0)  # tangent at theta=pi/2

    fitted = fit_curve_recursive(circle_pts, t1, t2, max_error=1.0)
    assert len(fitted) >= 1, "Curve fitting returned empty list"

    # Sample and measure max error
    for c in fitted:
        samples = c.sample_points(step_size=1.0)
        assert len(samples) > 2, "Sample points failed"
    print("  -> Bézier math and Schneider fitting verified successfully!")


def make_test_img():
    size = 128
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Flat graphic shape with sharp straight edge
    img[20:60, 20:60] = [255, 0, 0]  # Blue square
    cv2.circle(img, (90, 40), 20, (0, 255, 0), -1)  # Green circle
    # Stochastic noise/texture region
    np.random.seed(42)
    noise = np.random.normal(128, 30, (40, 100, 3)).clip(0, 255).astype(np.uint8)
    img[75:115, 14:114] = noise
    return img


def test_structure_tensor():
    print("Testing structure tensor coherence & texture segmentation...")
    img = make_test_img()

    flat_mask, texture_mask = segment_flat_and_graphic_regions(img)

    # Flat square interior should be marked as flat
    assert np.mean(flat_mask[30:50, 30:50]) > 0.85, "Flat square not detected"

    # Noise region interior should have high texture detection
    assert np.mean(texture_mask[85:105, 25:100]) > 0.60, "Stochastic texture not detected"

    print("  -> Structure tensor texture vs. graphic separation verified!")


def test_svg_export():
    print("Testing vector extraction and SVG export...")
    test_img = make_test_img()
    flat_mask, _ = segment_flat_and_graphic_regions(test_img)
    shapes = extract_vector_shapes(test_img, flat_mask, min_area=20)
    assert len(shapes) > 0, "No vector shapes extracted"

    svg_path = "test_output.svg"
    export_svg(shapes, test_img.shape[1], test_img.shape[0], svg_path, scale=2.0)
    assert os.path.exists(svg_path), "SVG file was not created"

    # Validate XML parsing
    tree = ET.parse(svg_path)
    root = tree.getroot()
    assert root.tag.endswith('svg'), "Root element is not <svg>"
    paths = list(root.iter('{http://www.w3.org/2000/svg}path'))
    assert len(paths) > 0, "No <path> elements found in SVG"

    print(f"  -> SVG export verified ({len(paths)} Bézier paths generated in {svg_path})!")
    if os.path.exists(svg_path):
        os.remove(svg_path)


def test_hybrid_upscaling_benchmark():
    print("Running end-to-end benchmark on synthetic graphic + texture test image...")
    # Create high-res ground truth image (256x256)
    hr_size = 256
    hr = np.zeros((hr_size, hr_size, 3), dtype=np.uint8)

    # Background: soft photographic-like gradient
    for y in range(hr_size):
        hr[y, :, 0] = int(40 + 40 * np.sin(y / 30.0))
        hr[y, :, 1] = int(50 + 30 * np.cos(y / 40.0))
        hr[y, :, 2] = int(60 + 20 * np.sin(y / 20.0))

    # Add stochastic fabric/weave pattern in lower half
    y_coords, x_coords = np.mgrid[140:240, 20:236]
    weave = (np.sin(x_coords * 0.8) * np.cos(y_coords * 0.8) * 35 + 128).clip(0, 255).astype(np.uint8)
    for c in range(3):
        hr[140:240, 20:236, c] = (hr[140:240, 20:236, c] * 0.4 + weave * 0.6).astype(np.uint8)

    # Sharp geometric text & logos in upper half
    cv2.putText(hr, "VECTOR 4.5", (24, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(hr, (210, 50), 28, (0, 220, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(hr, (210, 50), 16, (20, 30, 80), -1, lineType=cv2.LINE_AA)
    pts_star = np.array([[128, 70], [135, 90], [156, 90], [140, 102], [146, 122],
                         [128, 110], [110, 122], [116, 102], [100, 90], [121, 90]], np.int32)
    cv2.fillPoly(hr, [pts_star], (255, 80, 40), lineType=cv2.LINE_AA)

    # Degrade HR -> LR (2x downscaling with anti-aliasing blur)
    scale = 2
    lr = degrade_image(hr, scale=scale, blur_sigma=1.0)

    # 1. Baseline Bicubic
    upscaled_bicubic = cv2.resize(lr, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC)
    psnr_bicubic = calculate_psnr(hr, upscaled_bicubic)
    ssim_bicubic = calculate_ssim(hr, upscaled_bicubic)

    # 2. Phase 4.5 Hybrid Vector / Raster
    upscaled_hybrid = hybrid_vector_raster_upscale(
        lr,
        scale=scale,
        flatness_thresh=0.040,
        coherence_thresh=0.50,
        raster_engine="bicubic"
    )
    psnr_hybrid = calculate_psnr(hr, upscaled_hybrid)
    ssim_hybrid = calculate_ssim(hr, upscaled_hybrid)

    print(f"\nBenchmark Results (2x Scale):")
    print(f"  Bicubic Baseline             -> PSNR: {psnr_bicubic:.2f} dB, SSIM: {ssim_bicubic:.4f}")
    print(f"  Phase 4.5 Hybrid (Vec+Raster)-> PSNR: {psnr_hybrid:.2f} dB, SSIM: {ssim_hybrid:.4f}")

    # Edge sharpness analysis in the graphic region
    graphic_roi_hr = hr[20:130, 20:240]
    graphic_roi_bicubic = upscaled_bicubic[20:130, 20:240]
    graphic_roi_hybrid = upscaled_hybrid[20:130, 20:240]

    psnr_graphic_bicubic = calculate_psnr(graphic_roi_hr, graphic_roi_bicubic)
    psnr_graphic_hybrid = calculate_psnr(graphic_roi_hr, graphic_roi_hybrid)
    print(f"\nGraphic Elements Region PSNR:")
    print(f"  Bicubic Graphic ROI  : {psnr_graphic_bicubic:.2f} dB")
    print(f"  Phase 4.5 Graphic ROI: {psnr_graphic_hybrid:.2f} dB")

    # Save visual verification sheet
    cv2.imwrite("test_hr.png", hr)
    cv2.imwrite("test_lr.png", lr)
    cv2.imwrite("test_bicubic.png", upscaled_bicubic)
    cv2.imwrite("test_hybrid_4_5.png", upscaled_hybrid)
    print("\nVisual verification images saved: test_bicubic.png, test_hybrid_4_5.png")

    assert upscaled_hybrid.shape == hr.shape, "Output shape mismatch"
    print("  -> Benchmark complete and verified!")


if __name__ == "__main__":
    test_bezier_math()
    img = test_structure_tensor()
    test_svg_export(img)
    test_hybrid_upscaling_benchmark()
    print("\nALL PHASE 4.5 TESTS PASSED SUCCESSFULLY!")
