"""
vector_raster_hybrid.py — Phase 4.5: Hybrid Vector / Raster Decomposition.

Decomposes an input image into:
  1. Vector Layer: Flat-color / graphic regions, sharp geometric contours,
     text, and logo boundaries, vectorized into continuous cubic Bézier splines.
  2. Raster Layer: Stochastic textures (skin pores, fabric weave, foliage,
     sensor noise), preserved and upscaled via continuous raster super-resolution.
  3. Resolution-Independent Recombination: Re-rasterizing the vector paths at
     arbitrary scale S (e.g. 2x, 4x, 8x) with subpixel anti-aliasing, seamlessly
     fused with the raster upscaler via an edge-aware transition mask.

API: numpy array in -> numpy array out (compliant with Phase 0 evaluation harness).
Also supports direct W3C SVG vector export.
"""

import os
import sys
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


# ============================================================================
# 1. Mathematical Primitives & Cubic Bézier Spline Fitting (Schneider's Algo)
# ============================================================================

class Point2D:
    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Point2D({self.x:.2f}, {self.y:.2f})"

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def dot(self, other: 'Point2D') -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> 'Point2D':
        l = self.length()
        if l < 1e-9:
            return Point2D(0.0, 0.0)
        return Point2D(self.x / l, self.y / l)

    def distance_to(self, other: 'Point2D') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class CubicBezier:
    """Parametric cubic Bézier curve segment: B(t) for t in [0, 1]."""
    __slots__ = ('p0', 'p1', 'p2', 'p3')

    def __init__(self, p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def eval(self, t: float) -> Point2D:
        """Evaluate curve at parameter t in [0, 1] using Bernstein polynomials."""
        t2 = t * t
        t3 = t2 * t
        u = 1.0 - t
        u2 = u * u
        u3 = u2 * u

        x = u3 * self.p0.x + 3.0 * u2 * t * self.p1.x + 3.0 * u * t2 * self.p2.x + t3 * self.p3.x
        y = u3 * self.p0.y + 3.0 * u2 * t * self.p1.y + 3.0 * u * t2 * self.p2.y + t3 * self.p3.y
        return Point2D(x, y)

    def scale(self, factor: float, center_align: bool = False) -> 'CubicBezier':
        """
        Scale curve parameters analytically to arbitrary resolution.
        If center_align is True, uses half-pixel alignment: (x + 0.5) * factor - 0.5.
        """
        if center_align:
            def tx(p: Point2D) -> Point2D:
                return Point2D((p.x + 0.5) * factor - 0.5, (p.y + 0.5) * factor - 0.5)
            return CubicBezier(tx(self.p0), tx(self.p1), tx(self.p2), tx(self.p3))
        else:
            return CubicBezier(self.p0 * factor, self.p1 * factor, self.p2 * factor, self.p3 * factor)

    def sample_points(self, step_size: float = 0.5) -> List[Point2D]:
        """Sample points along curve so distance between samples is ~step_size."""
        chord = (self.p3 - self.p0).length()
        poly_len = (self.p1 - self.p0).length() + (self.p2 - self.p1).length() + (self.p3 - self.p2).length()
        approx_len = (chord + poly_len) * 0.5
        num_samples = max(4, int(math.ceil(approx_len / max(0.1, step_size))))

        points = []
        for i in range(num_samples + 1):
            t = i / float(num_samples)
            points.append(self.eval(t))
        return points


def fit_cubic_bezier_segment(points: List[Point2D], t_hat1: Point2D, t_hat2: Point2D) -> CubicBezier:
    """
    Fits a single cubic Bézier segment through points with specified end tangents
    using linear least squares (Schneider's method).
    """
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    if n <= 2:
        dist = p0.distance_to(p3) / 3.0
        return CubicBezier(p0, p0 + t_hat1 * dist, p3 + t_hat2 * dist, p3)

    # Chord length parameterization
    u = [0.0] * n
    for i in range(1, n):
        u[i] = u[i - 1] + points[i].distance_to(points[i - 1])
    total_len = u[-1]
    if total_len < 1e-9:
        return CubicBezier(p0, p0, p3, p3)

    for i in range(1, n):
        u[i] /= total_len

    c00 = 0.0
    c01 = 0.0
    c11 = 0.0
    x0 = 0.0
    x1 = 0.0

    for i in range(n):
        ui = u[i]
        u2 = ui * ui
        u3 = u2 * ui
        om_u = 1.0 - ui
        om_u2 = om_u * om_u
        om_u3 = om_u2 * om_u

        a1 = t_hat1 * (3.0 * om_u2 * ui)
        a2 = t_hat2 * (3.0 * om_u * u2)

        c00 += a1.dot(a1)
        c01 += a1.dot(a2)
        c11 += a2.dot(a2)

        b0_b3_part = p0 * (om_u3 + 3.0 * om_u2 * ui) + p3 * (3.0 * om_u * u2 + u3)
        diff = points[i] - b0_b3_part

        x0 += a1.dot(diff)
        x1 += a2.dot(diff)

    det = c00 * c11 - c01 * c01
    dist = total_len / 3.0

    if abs(det) > 1e-9:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det
        if alpha1 > 0 and alpha2 > 0 and alpha1 < total_len * 2 and alpha2 < total_len * 2:
            return CubicBezier(p0, p0 + t_hat1 * alpha1, p3 + t_hat2 * alpha2, p3)

    return CubicBezier(p0, p0 + t_hat1 * dist, p3 + t_hat2 * dist, p3)


def fit_curve_recursive(points: List[Point2D], t_hat1: Point2D, t_hat2: Point2D,
                        max_error: float = 1.0) -> List[CubicBezier]:
    """
    Recursively fits cubic Bézier curves to points until maximum Euclidean error < max_error.
    Schneider's algorithm.
    """
    n = len(points)
    if n <= 2:
        dist = points[0].distance_to(points[-1]) / 3.0
        return [CubicBezier(points[0], points[0] + t_hat1 * dist, points[-1] + t_hat2 * dist, points[-1])]

    curve = fit_cubic_bezier_segment(points, t_hat1, t_hat2)

    u = [0.0] * n
    for i in range(1, n):
        u[i] = u[i - 1] + points[i].distance_to(points[i - 1])
    total_len = u[-1]
    if total_len > 1e-9:
        for i in range(1, n):
            u[i] /= total_len

    max_dist = 0.0
    split_idx = n // 2
    for i in range(1, n - 1):
        pt_on_curve = curve.eval(u[i])
        dist = pt_on_curve.distance_to(points[i])
        if dist > max_dist:
            max_dist = dist
            split_idx = i

    if max_dist <= max_error or n <= 4:
        return [curve]

    center_pt = points[split_idx]
    v_prev = (center_pt - points[split_idx - 1]).normalized()
    v_next = (points[split_idx + 1] - center_pt).normalized()
    t_center = (v_prev + v_next).normalized()
    if t_center.length() < 1e-6:
        t_center = v_prev

    left_curves = fit_curve_recursive(points[:split_idx + 1], t_hat1, t_center * (-1.0), max_error)
    right_curves = fit_curve_recursive(points[split_idx:], t_center, t_hat2, max_error)
    return left_curves + right_curves


def vectorize_contour(contour: np.ndarray, epsilon: float = 0.5,
                      max_bezier_error: float = 0.8, corner_angle_thresh: float = 135.0) -> List[CubicBezier]:
    """
    Converts a discrete OpenCV contour into an analytical set of Cubic Bézier curves.
    Preserves sharp corners by segmenting at acute angles before spline fitting.
    """
    pts = contour.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return []

    # Douglas-Peucker polygon approximation to reduce pixel digitization noise
    approx = cv2.approxPolyDP(pts.astype(np.float32), epsilon=epsilon, closed=True).squeeze()
    if approx.ndim != 2 or len(approx) < 3:
        approx = pts

    point_objs = [Point2D(p[0], p[1]) for p in approx]
    m = len(point_objs)

    # Detect sharp corner indices
    corner_indices = []
    angle_rad_thresh = math.radians(corner_angle_thresh)

    for i in range(m):
        p_prev = point_objs[(i - 1) % m]
        p_curr = point_objs[i]
        p_next = point_objs[(i + 1) % m]

        v1 = (p_prev - p_curr).normalized()
        v2 = (p_next - p_curr).normalized()
        dot = max(-1.0, min(1.0, v1.dot(v2)))
        angle = math.acos(dot)

        if angle < angle_rad_thresh:
            corner_indices.append(i)

    if not corner_indices:
        corner_indices = [0]

    all_curves = []
    num_corners = len(corner_indices)

    for k in range(num_corners):
        idx_start = corner_indices[k]
        idx_end = corner_indices[(k + 1) % num_corners]

        if idx_end > idx_start:
            seg_points = point_objs[idx_start:idx_end + 1]
        else:
            seg_points = point_objs[idx_start:] + point_objs[:idx_end + 1]

        if len(seg_points) < 2:
            continue

        t_start = (seg_points[1] - seg_points[0]).normalized()
        t_end = (seg_points[-2] - seg_points[-1]).normalized()

        curves = fit_curve_recursive(seg_points, t_start, t_end, max_error=max_bezier_error)
        all_curves.extend(curves)

    return all_curves


# ============================================================================
# 2. Structure Tensor & Texture / Graphic Region Segmentation
# ============================================================================

def compute_structure_tensor(gray: np.ndarray, sigma: float = 1.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes structure tensor components Jxx, Jyy, Jxy of an image.
    gray: float32 in [0, 1]
    Returns: (coherence, trace_energy, local_std)
    """
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    ksize = int(math.ceil(sigma * 3) * 2 + 1)
    jxx = cv2.GaussianBlur(gx * gx, (ksize, ksize), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (ksize, ksize), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (ksize, ksize), sigma)

    diff = jxx - jyy
    sqrt_term = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * jxy * jxy))
    lambda1 = 0.5 * (jxx + jyy + sqrt_term)
    lambda2 = 0.5 * (jxx + jyy - sqrt_term)

    # Coherence (anisotropy) C in [0, 1]
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    trace_energy = lambda1 + lambda2

    # Local variance / standard deviation
    mean_i = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    mean_i2 = cv2.GaussianBlur(gray * gray, (ksize, ksize), sigma)
    local_var = np.maximum(0.0, mean_i2 - mean_i * mean_i)
    local_std = np.sqrt(local_var)

    return coherence, trace_energy, local_std


def segment_flat_and_graphic_regions(
    img: np.ndarray,
    flatness_std_thresh: float = 0.045,
    coherence_thresh: float = 0.55,
    energy_edge_thresh: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments the image into:
      flat_mask: boolean mask where True indicates flat graphic regions or sharp edges.
      texture_mask: boolean mask where True indicates stochastic texture (skin, fabric, noise).
    """
    img_f = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.copy()
    if img_f.ndim == 3:
        gray = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        gray = img_f

    coherence, trace_energy, local_std = compute_structure_tensor(gray, sigma=1.2)

    is_flat_color = (local_std < flatness_std_thresh)
    is_coherent_edge = (coherence > coherence_thresh) & (trace_energy > energy_edge_thresh)

    raw_graphic = is_flat_color | is_coherent_edge

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_graphic = cv2.morphologyEx(raw_graphic.astype(np.uint8), cv2.MORPH_CLOSE, kernel_3)
    cleaned_graphic = cv2.morphologyEx(cleaned_graphic, cv2.MORPH_OPEN, kernel_3)

    flat_mask = (cleaned_graphic > 0)
    texture_mask = ~flat_mask

    return flat_mask, texture_mask


# ============================================================================
# 3. Vector Shape Extraction & Resolution-Independent Rendering
# ============================================================================

class VectorShape:
    """Represents an extracted, vectorized region with Bézier boundary and color."""
    __slots__ = ('curves', 'fill_color', 'is_hole', 'area', 'bbox')

    def __init__(self, curves: List[CubicBezier], fill_color: Tuple[int, int, int],
                 is_hole: bool = False, area: float = 0.0, bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        self.curves = curves
        self.fill_color = fill_color
        self.is_hole = is_hole
        self.area = area
        self.bbox = bbox


def extract_vector_shapes(img: np.ndarray, flat_mask: np.ndarray,
                          min_area: int = 8, max_bezier_error: float = 0.8,
                          num_clusters: int = 8, min_contrast: float = 50.0) -> List[VectorShape]:
    """
    Extracts closed vector contours and fits Bézier splines for all distinct graphic elements.
    Uses bilateral edge-preserving filtering + color quantization to identify individual shapes
    and their internal holes without wiping out the canvas.
    """
    h, w = img.shape[:2]
    img_bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Edge-preserving filter to smooth within regions while preserving step edges
    smooth = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=40, sigmaSpace=40)

    # Masked color quantization within flat/graphic regions
    flat_pixels = smooth[flat_mask]
    if len(flat_pixels) < 20:
        return []

    k_clusters = min(num_clusters, max(2, len(flat_pixels) // 50))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    _, labels, centers = cv2.kmeans(
        flat_pixels.astype(np.float32), k_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # Reconstruct label map for flat regions
    label_img = np.full((h, w), -1, dtype=np.int32)
    label_img[flat_mask] = labels.flatten()
    total_pixels = h * w

    shapes = []

    for k in range(k_clusters):
        mask_k = (label_img == k).astype(np.uint8) * 255
        pixel_count = np.sum(mask_k > 0)

        # If a single color cluster covers more than 40% of the entire image,
        # it is the background substrate; skip rendering it as a solid foreground shape
        if pixel_count / total_pixels > 0.40:
            continue

        contours, hierarchy = cv2.findContours(mask_k, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            continue

        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area or area > (total_pixels * 0.35):
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            is_hole = (hierarchy[i][3] != -1)

            # Sample interior color
            mask_cnt = np.zeros((ch, cw), dtype=np.uint8)
            cnt_shifted = cnt - np.array([[[x, y]]])
            cv2.drawContours(mask_cnt, [cnt_shifted], -1, 255, -1)

            roi_img = img_bgr[y:y+ch, x:x+cw]
            pts = roi_img[mask_cnt > 0]
            if len(pts) == 0:
                continue

            # Check interior standard deviation: reject stochastic texture patches
            std = np.std(pts, axis=0).mean()
            if std > 28.0:
                continue

            median_col = np.median(pts, axis=0).astype(int)
            fill_color = (int(median_col[0]), int(median_col[1]), int(median_col[2]))

            # Check boundary contrast against surrounding background to reject gentle skin gradients
            mask_outer = cv2.dilate(mask_cnt, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
            mask_ring = cv2.bitwise_xor(mask_outer, mask_cnt)
            ring_pts = roi_img[mask_ring > 0]
            if len(ring_pts) > 0:
                outer_col = np.median(ring_pts, axis=0)
                contrast = np.linalg.norm(median_col - outer_col)
                if contrast < min_contrast:
                    continue

            # Photographic protection gates:
            # 1. Reject human skin patches from being vectorized into solid flat polygons
            ycrcb_col = cv2.cvtColor(np.uint8([[fill_color]]), cv2.COLOR_BGR2YCrCb)[0, 0]
            cr_val, cb_val = int(ycrcb_col[1]), int(ycrcb_col[2])
            hsv_col = cv2.cvtColor(np.uint8([[fill_color]]), cv2.COLOR_BGR2HSV)[0, 0]
            hue_val, sat_val, val_val = int(hsv_col[0]), int(hsv_col[1]), int(hsv_col[2])
            is_skin_tone = (133 <= cr_val <= 175) and (77 <= cb_val <= 130) and (sat_val < 160) and (hue_val < 20)
            if is_skin_tone:
                continue

            # 2. Reject natural photographic shadows (very low luminance without being dark text)
            if val_val < 45:
                continue

            # 3. Reject large background regions (graphic typography/badges are compact symbols in LR)
            if area > (total_pixels * 0.15):
                continue

            # Vectorize contour into analytical cubic Bézier curves
            curves = vectorize_contour(cnt, epsilon=0.4, max_bezier_error=max_bezier_error)
            if curves:
                shapes.append(VectorShape(
                    curves=curves,
                    fill_color=fill_color,
                    is_hole=is_hole,
                    area=area,
                    bbox=(x, y, cw, ch)
                ))

    # Sort shapes: outer shapes (is_hole=False) by area descending,
    # then holes on top to maintain proper z-indexing
    shapes.sort(key=lambda s: (s.is_hole, -s.area))
    return shapes


def render_vector_shapes(shapes: List[VectorShape], out_w: int, out_h: int,
                         scale: float, base_canvas: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Renders vector shapes at target resolution (out_w, out_h) with subpixel anti-aliasing.
    Scale factor is applied analytically to continuous Bézier parameters with half-pixel alignment.
    """
    if base_canvas is not None:
        canvas = base_canvas.copy()
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for shape in shapes:
        if not shape.curves:
            continue

        # Scale Bézier curves analytically with half-pixel alignment
        scaled_curves = [c.scale(scale, center_align=True) for c in shape.curves]

        # Sample continuous curves at dense subpixel points
        poly_points = []
        for c in scaled_curves:
            pts = c.sample_points(step_size=0.5)
            for p in pts[:-1]:
                poly_points.append([p.x, p.y])

        if len(poly_points) < 3:
            continue

        poly_arr = np.array(poly_points, dtype=np.float32)

        # Subpixel fixed-point drawing with 16-subpixel precision
        shift = 4
        subpixel_scale = 1 << shift
        subpixel_pts = np.round(poly_arr * subpixel_scale).astype(np.int32)

        color = (shape.fill_color[0], shape.fill_color[1], shape.fill_color[2])
        cv2.fillPoly(canvas, [subpixel_pts], color, lineType=cv2.LINE_AA, shift=shift)

    return canvas


# ============================================================================
# 4. Direct SVG Vector Exporter
# ============================================================================

def export_svg(shapes: List[VectorShape], width: int, height: int, filepath: str,
               scale: float = 1.0) -> None:
    """
    Exports extracted Bézier vector shapes directly to a standard W3C SVG file.
    Opens in any browser, Figma, Illustrator, or Inkscape at infinite resolution!
    """
    svg_w = int(width * scale)
    svg_h = int(height * scale)

    lines = [
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">'
    ]

    for shape in shapes:
        if not shape.curves:
            continue

        scaled_curves = [c.scale(scale, center_align=False) for c in shape.curves]
        p0 = scaled_curves[0].p0
        path_d = [f"M {p0.x:.2f} {p0.y:.2f}"]

        for c in scaled_curves:
            path_d.append(f"C {c.p1.x:.2f} {c.p1.y:.2f}, {c.p2.x:.2f} {c.p2.y:.2f}, {c.p3.x:.2f} {c.p3.y:.2f}")

        path_d.append("Z")
        d_str = " ".join(path_d)

        b, g, r = shape.fill_color
        color_hex = f"#{r:02x}{g:02x}{b:02x}"

        lines.append(f'  <path d="{d_str}" fill="{color_hex}" fill-rule="evenodd" stroke="{color_hex}" stroke-width="0.5"/>')

    lines.append('</svg>\n')

    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ============================================================================
# 5. Hybrid Vector / Raster Recombination (Fusion Engine)
# ============================================================================

def hybrid_vector_raster_upscale(
    lr_img: np.ndarray,
    scale: int = 4,
    flatness_thresh: float = 0.045,
    coherence_thresh: float = 0.55,
    max_bezier_error: float = 0.8,
    min_contrast: float = 50.0,
    raster_engine: str = "bicubic",
    export_svg_path: Optional[str] = None
) -> np.ndarray:
    """
    Phase 4.5 Hybrid Vector/Raster Upscaling:
    1. Segments LR image into flat/graphic zones vs. stochastic textures.
    2. Vectorizes graphic contours into cubic Bézier splines (resolution-independent).
    3. Renders vector paths at target resolution S with subpixel anti-aliasing.
    4. Upscales base/stochastic textures via raster super-resolution engine.
    5. Seamlessly recombines using an edge-aware transition mask.

    Parameters:
      lr_img: uint8 or float32 image (H, W, 3) or (H, W)
      scale: upscaling factor (2, 4, 8, etc.)
      flatness_thresh: local std threshold to detect flat regions
      coherence_thresh: structure tensor coherence threshold for sharp edges
      max_bezier_error: maximum allowed pixel fitting error for Bézier splines
      min_contrast: minimum boundary contrast to qualify as a graphic element
      raster_engine: "bicubic", "lanczos", "tv", or custom callable
      export_svg_path: optional path to save true SVG file
    Returns:
      Upscaled image at (H*scale, W*scale) with infinite edge sharpness.
    """
    is_uint8 = (lr_img.dtype == np.uint8)
    lr_uint8 = lr_img if is_uint8 else np.clip(lr_img * 255.0, 0, 255).astype(np.uint8)

    is_mono = (lr_uint8.ndim == 2) or (lr_uint8.shape[2] == 1)
    if is_mono:
        lr_bgr = cv2.cvtColor(lr_uint8, cv2.COLOR_GRAY2BGR)
    else:
        lr_bgr = lr_uint8

    h, w = lr_bgr.shape[:2]
    out_h, out_w = h * scale, w * scale

    # Step 1: Detect flat/graphic regions vs. stochastic textures
    flat_mask, texture_mask = segment_flat_and_graphic_regions(
        lr_bgr,
        flatness_std_thresh=flatness_thresh,
        coherence_thresh=coherence_thresh
    )

    # Step 2: Extract & vectorize high-contrast graphic contours into Bézier splines
    shapes = extract_vector_shapes(
        lr_bgr,
        flat_mask=flat_mask,
        min_area=8,
        max_bezier_error=max_bezier_error,
        min_contrast=min_contrast
    )

    # Optional SVG export
    if export_svg_path:
        export_svg(shapes, w, h, export_svg_path, scale=1.0)

    # Step 3: Raster upscaling of base / texture layer
    if raster_engine == "lanczos":
        raster_hr = cv2.resize(lr_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    elif raster_engine == "tv":
        from tv_refinement import tv_super_resolution_refine
        base_bicubic = cv2.resize(lr_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        raster_hr = tv_super_resolution_refine(lr_bgr, base_bicubic, scale=scale, lambda_tv=0.015, num_iters=4)
    elif callable(raster_engine):
        raster_hr = raster_engine(lr_bgr, scale)
    else:
        raster_hr = cv2.resize(lr_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # If no vector shapes detected, return raster HR directly
    if not shapes:
        out_bgr = raster_hr
    else:
        # Step 4: Render vector shapes on top of raster HR with subpixel anti-aliasing
        vector_canvas = render_vector_shapes(shapes, out_w, out_h, scale=float(scale), base_canvas=raster_hr)

        # Step 5: Compute high-resolution vector presence mask
        # Difference between vector canvas and raster HR identifies rendered vector pixels
        diff = np.abs(vector_canvas.astype(np.float32) - raster_hr.astype(np.float32)).max(axis=2)
        vec_presence = (diff > 5.0).astype(np.float32)

        # Feather edges slightly (1-2 pixels) for seamless anti-aliased blend
        feather_size = max(3, int(scale) | 1)
        alpha_map = cv2.GaussianBlur(vec_presence, (feather_size, feather_size), 0)[:, :, np.newaxis]
        alpha_map = np.clip(alpha_map * 1.1, 0.0, 1.0)

        blended = vector_canvas.astype(np.float32) * alpha_map + raster_hr.astype(np.float32) * (1.0 - alpha_map)
        out_bgr = np.clip(blended, 0, 255).astype(np.uint8)

    if is_mono:
        out_res = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
    else:
        out_res = out_bgr

    if not is_uint8:
        return out_res.astype(np.float32) / 255.0
    return out_res


# ============================================================================
# 6. Standalone CLI & Self-Test
# ============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python vector_raster_hybrid.py <input_image> <output_image> [--scale N] [--svg output.svg] [--engine bicubic|lanczos|tv]")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    scale = 4
    engine = "bicubic"
    svg_path = None

    for i, arg in enumerate(sys.argv):
        if arg == "--scale" and i + 1 < len(sys.argv):
            scale = int(sys.argv[i + 1])
        elif arg == "--engine" and i + 1 < len(sys.argv):
            engine = sys.argv[i + 1]
        elif arg == "--svg" and i + 1 < len(sys.argv):
            svg_path = sys.argv[i + 1]

    if not os.path.exists(input_path):
        print(f"Error: input image not found at {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read image {input_path}")
        return

    print(f"Running Phase 4.5 Hybrid Vector / Raster Decomposition on {input_path} ({scale}x)...")
    out = hybrid_vector_raster_upscale(
        img,
        scale=scale,
        raster_engine=engine,
        export_svg_path=svg_path
    )

    cv2.imwrite(output_path, out)
    print(f"Saved upscaled image to {output_path}")
    if svg_path:
        print(f"Saved resolution-independent SVG to {svg_path}")


if __name__ == "__main__":
    main()
