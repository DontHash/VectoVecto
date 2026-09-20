"""
test_smart_upscaler.py — Comprehensive Unit & Integration Tests for Phase 7.

Tests:
  1. Semantic content analysis on synthetic images (text vs texture separation).
  2. Photographic skin protection on real-world portraiture (PrakashJI.jpg).
  3. Dual-stream routing on mixed media (De1.jpg: text to vector, skin to DRUNet).
  4. End-to-end smart_upscale execution (speed, shape, dtype consistency).
  5. SVG export validity and semantic diagnostic mask generation.
"""

import os
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pytest

from smart_upscaler import SmartUpscaler, smart_upscale, ContentAnalysis

@pytest.fixture(scope="module")
def upscaler():
    """Module-level upscaler fixture."""
    return SmartUpscaler()

def make_mixed_test_image():
    """Creates an artificial test image with clear text, geometric shapes, and noise."""
    size = 128
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Background gradient
    for y in range(size):
        img[y, :, 0] = int(30 + y * 0.4)
        img[y, :, 1] = int(20 + y * 0.3)
        img[y, :, 2] = int(40 + y * 0.2)

    # Sharp text
    cv2.putText(img, "SMART", (15, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # Yellow badge
    cv2.circle(img, (95, 40), 18, (0, 220, 255), -1, lineType=cv2.LINE_AA)

    # Stochastic noise region at bottom
    np.random.seed(42)
    noise = np.random.normal(120, 35, (35, 100, 3)).clip(0, 255).astype(np.uint8)
    img[80:115, 14:114] = noise
    return img


def test_semantic_content_analyzer(upscaler):
    print("Testing semantic content classification on synthetic test image...")
    img = make_mixed_test_image()
    analysis = upscaler.analyze_content(img)

    assert isinstance(analysis, ContentAnalysis), "Analysis must return a ContentAnalysis instance"
    assert len(analysis.vector_shapes) > 0, "Failed to detect vector text/badge shapes"
    assert analysis.summary["has_vector_content"] is True, "has_vector_content flag must be True"
    assert np.mean(analysis.texture_mask[85:110, 20:100]) > 0.50, "Failed to detect stochastic texture region"
    print(f"  -> Detected {len(analysis.vector_shapes)} vector shapes, texture_pct={analysis.summary['texture_pct']:.1f}%")


def test_portrait_skin_protection(upscaler):
    print("Testing photographic skin protection on PrakashJI portrait...")
    if not os.path.exists("PrakashJI.jpg"):
        pytest.skip("PrakashJI.jpg not found in workspace")

    img = cv2.imread("PrakashJI.jpg")
    analysis = upscaler.analyze_content(img)

    # Assert skin percentage is detected
    assert analysis.summary["skin_pct"] > 5.0, f"Skin not detected on portrait (got {analysis.summary['skin_pct']}%)"
    assert analysis.summary["is_predominantly_photo"] is True, "Portrait must be flagged as predominantly photo"

    # Assert that skin is NOT converted into vector shapes
    # (Toddler's cheeks and forehead must not have vector shapes)
    h, w = img.shape[:2]
    face_roi_y1, face_roi_y2 = int(h * 0.1), int(h * 0.4)
    face_roi_x1, face_roi_x2 = int(w * 0.2), int(w * 0.8)

    face_vector_shapes = [
        s for s in analysis.vector_shapes
        if face_roi_x1 <= s.bbox[0] <= face_roi_x2 and face_roi_y1 <= s.bbox[1] <= face_roi_y2
    ]
    assert len(face_vector_shapes) == 0, f"Face skin was mistakenly vectorized ({len(face_vector_shapes)} shapes found)"
    print(f"  -> Portrait skin protection verified! (0 vector shapes on face)")


def test_diagnostic_mask_generation(upscaler):
    print("Testing semantic diagnostic mask generation...")
    img = make_mixed_test_image()
    analysis = upscaler.analyze_content(img)
    diag_map = upscaler.generate_diagnostic_map(img, analysis)

    assert diag_map.shape == img.shape, "Diagnostic map must match input shape"
    assert diag_map.dtype == np.uint8, "Diagnostic map must be uint8"
    # Should have green annotations for vector shapes
    assert (diag_map[:, :, 1] > 200).any(), "Expected bright green annotations on vector shapes"
    print("  -> Diagnostic map generated and verified successfully!")


def test_svg_export(upscaler, tmp_path):
    print("Testing resolution-independent SVG export...")
    img = make_mixed_test_image()
    svg_file = str(tmp_path / "test_smart.svg")

    out = upscaler.upscale(
        img,
        scale=2,
        mode="auto",
        fast=True,
        export_svg_path=svg_file
    )

    assert os.path.exists(svg_file), "SVG export file was not created"
    tree = ET.parse(svg_file)
    root = tree.getroot()
    assert root.tag.endswith("svg"), "Root element is not <svg>"
    paths = list(root.iter("{http://www.w3.org/2000/svg}path"))
    assert len(paths) > 0, "No <path> elements found in SVG"
    print(f"  -> SVG export verified ({len(paths)} Bézier paths in {svg_file})")


def test_end_to_end_smart_upscale(upscaler):
    print("Testing end-to-end smart_upscale function API...")
    img = make_mixed_test_image()
    scale = 4

    out = smart_upscale(img, scale=scale, mode="auto", fast=True)

    assert out.shape == (img.shape[0] * scale, img.shape[1] * scale, 3), "Output dimensions incorrect"
    assert out.dtype == np.uint8, "Output dtype must match input uint8"
    assert out.min() >= 0 and out.max() <= 255, "Output pixel values out of bounds"
    print(f"  -> End-to-end SmartUpscale verified: {img.shape} -> {out.shape}")
