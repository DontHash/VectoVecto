"""
smart_upscaler.py — Phase 7 v2: professional routing pipeline ("Smart Router").

What changed in v2 (the anti-clay release):
  * The neural core is now `sr_engine` — a fine-tuned Real-ESRGAN-class model
    (artifacts/tier_c) with x4plus / ONNX / ncnn fallbacks, instead of the tiny
    1.5M deep-unfolding DRUNet trained on Gaussian-only degradation.
  * The always-on unsharp mask and forced micro-grain were removed. Grain is
    opt-in (`grain_strength=0` by default). These were band-aids that produced
    the "clay / plastic" sheen users noticed.
  * `mode="fidelity"` keeps the paper-style deep-unfolding path (TV-free,
    reconstruction-consistent, hallucination-free) as a distinct choice.

Content routing (unchanged):
  1. High-contrast typography/logos -> Bézier vector engine (SVG export).
  2. Photographic content             -> professional neural SR model.
  3. Stochastic texture               -> preserved by the same neural model.
  4. Flat/planar substrates           -> edge-aware subpixel blending.

API: numpy array in -> numpy array out (Phase 0 harness contract).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import sr_engine
from sr_engine import load_engine, tiled_run

import vector_raster_hybrid as vrh  # noqa: F401  (kept for API compatibility)
from vector_raster_hybrid import (
    VectorShape,
    compute_structure_tensor,  # noqa: F401
    segment_flat_and_graphic_regions,
    extract_vector_shapes,
    render_vector_shapes,
    export_svg,
)

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts", "deep_sr", "best_checkpoint.pth"
)


# ---------------------------------------------------------------------------
# cheap page classifier (document vs photo) — no models, no OCR
# ---------------------------------------------------------------------------

def _skin_ratio(img_bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 130)
    return float(mask.mean())


def page_likeness(img_bgr: np.ndarray) -> float:
    """How much does this look like a document page? [0..1], higher = page.

    Signals: low color saturation, many text-sized connected components,
    components aligned in rows, little skin. Deliberately biased toward
    "document" when uncertain (documents are the product).
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1000:
        f = 1000.0 / max(h, w)
        img = cv2.resize(img_bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean()) / 255.0
    skin = _skin_ratio(img)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    total = float(img.shape[0] * img.shape[1])
    comps = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if 3 <= hh <= max(60, img.shape[0] // 20) and 2 <= ww <= 400 and area >= 6:
            if area / max(ww * hh, 1) > 0.08:
                patch = gray[y:y + hh, x:x + ww]
                # real text is high-contrast ink; blurred texture blobs are not
                spread = float(np.percentile(patch, 90) - np.percentile(patch, 10))
                if spread >= 70.0:
                    comps.append((x, y, ww, hh, area))
    if not comps:
        return 0.0
    density = sum(c[4] for c in comps) / total
    heights = sorted(c[3] for c in comps)
    med_h = heights[len(heights) // 2]

    rows = {}
    for _x, y, _ww, hh, _area in comps:
        key = int((y + hh / 2) / max(med_h * 1.4, 1))
        rows.setdefault(key, 0)
        rows[key] += 1
    in_rows = sum(v for v in rows.values() if v >= 3)
    row_frac = in_rows / len(comps)

    low_sat = 1.0 - min(sat * 3.0, 1.0)
    text_energy = min(density * 25.0, 1.0)
    score = 0.45 * low_sat + 0.35 * text_energy + 0.20 * row_frac - 1.2 * skin
    return float(np.clip(score, 0.0, 1.0))


def is_document(img_bgr: np.ndarray, threshold: float = 0.45) -> bool:
    return page_likeness(img_bgr) >= threshold


class ContentAnalysis:
    """Encapsulates the semantic decomposition of an image into content domains."""

    def __init__(self, vector_shapes: List[VectorShape], skin_mask: np.ndarray,
                 flat_mask: np.ndarray, texture_mask: np.ndarray,
                 summary: Dict[str, Any]):
        self.vector_shapes = vector_shapes
        self.skin_mask = skin_mask
        self.flat_mask = flat_mask
        self.texture_mask = texture_mask
        self.summary = summary

    def __repr__(self):
        return (f"<ContentAnalysis: {len(self.vector_shapes)} vector shapes, "
                f"skin={self.summary.get('skin_pct', 0):.1f}%, "
                f"texture={self.summary.get('texture_pct', 0):.1f}%>")


class SmartUpscaler:
    """Phase 7 meta-upscaler: routes photo content to a professional SR model and
    graphic content to the vector engine."""

    def __init__(self, checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None, model_spec: str = "auto",
                 tile: Optional[int] = None, tta: bool = False):
        self.model_spec = model_spec
        self.device = device or "auto"
        self.tile = tile
        self.tta = tta
        self.checkpoint_path = checkpoint_path or _DEFAULT_CHECKPOINT
        self.engine: Optional[sr_engine.Engine] = None
        self._fidelity_fn = None

    # ------------------------------------------------------------------
    # Neural engine management
    # ------------------------------------------------------------------
    def _init_engine(self):
        if self.engine is not None:
            return
        try:
            self.engine = load_engine(self.model_spec, scale=4, device=self.device,
                                      tile=self.tile, fp16=True, tta=self.tta)
            print(f"  [SmartUpscaler] engine ready: {self.engine}")
        except Exception as e:  # noqa: BLE001
            print(f"  [SmartUpscaler] engine load failed ({e}); using bicubic fallback")
            self.engine = sr_engine.ClassicalEngine("bicubic", 4)

    def _init_fidelity(self):
        if self._fidelity_fn is not None:
            return
        import torch
        from drunet import DRUNet
        from deep_unfolding import DeepUnfoldingSR, create_gaussian_kernel

        dev = torch.device("cuda" if (self.device in ("auto", "cuda") and torch.cuda.is_available())
                           else "cpu")
        sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        unfolding = DeepUnfoldingSR(DRUNet(in_channels=3, num_feat=64, num_blocks=20),
                                    iterations=5, scale=4)
        unfolding.load_state_dict(sd, strict=True)
        unfolding.eval().to(dev)
        kernel = create_gaussian_kernel(sigma=1.2).to(dev)

        def fn(patch_bgr: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                y = unfolding(x, kernel)
            y = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            return cv2.cvtColor((np.clip(y, 0, 1) * 255).round().astype(np.uint8),
                                cv2.COLOR_RGB2BGR)

        self._fidelity_fn = fn
        print("  [SmartUpscaler] fidelity engine ready (deep unfolding, TV-free)")

    # ------------------------------------------------------------------
    # Content analysis (unchanged behavior)
    # ------------------------------------------------------------------
    def analyze_content(self, img_bgr: np.ndarray) -> ContentAnalysis:
        h, w = img_bgr.shape[:2]
        total_pixels = h * w

        flat_mask, texture_mask = segment_flat_and_graphic_regions(img_bgr)

        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat, hue = hsv[:, :, 1], hsv[:, :, 0]

        skin_mask = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 130) & (sat < 160) & (hue < 22)
        skin_pct = float(np.sum(skin_mask) / total_pixels * 100.0)
        texture_pct = float(np.sum(texture_mask) / total_pixels * 100.0)

        shapes = extract_vector_shapes(img_bgr, flat_mask=flat_mask, min_area=8,
                                       max_bezier_error=0.8, min_contrast=50.0)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        max_area = min(2500, int(total_pixels * 0.025))
        is_predom_photo = (skin_pct > 8.0) or (texture_pct > 35.0)

        clean_shapes = []
        for s in shapes:
            sx, sy, sw, sh = s.bbox
            if sx <= 1 or sy <= 1 or (sx + sw >= w - 1) or (sy + sh >= h - 1):
                continue
            if s.area > max_area:
                continue
            if np.mean(skin_mask[sy:sy + sh, sx:sx + sw]) > 0.35:
                continue
            sub_grad = grad_mag[sy:sy + sh, sx:sx + sw]
            if sub_grad.max() < 120.0 or sub_grad.mean() < 30.0:
                continue
            hsv_col = cv2.cvtColor(np.uint8([[s.fill_color]]), cv2.COLOR_BGR2HSV)[0, 0]
            if is_predom_photo and int(hsv_col[1]) < 70:
                continue
            clean_shapes.append(s)

        summary = {
            "total_shapes": len(clean_shapes),
            "skin_pct": skin_pct,
            "texture_pct": texture_pct,
            "has_vector_content": len(clean_shapes) > 0,
            "is_predominantly_photo": is_predom_photo,
        }
        return ContentAnalysis(clean_shapes, skin_mask, flat_mask, texture_mask, summary)

    def generate_diagnostic_map(self, img_bgr: np.ndarray,
                                analysis: ContentAnalysis) -> np.ndarray:
        diag = img_bgr.astype(np.float32) * 0.45
        tex_idx = analysis.texture_mask > 0
        diag[tex_idx, 0] += 40
        diag[tex_idx, 2] += 50
        skin_idx = analysis.skin_mask > 0
        diag[skin_idx, 0] += 70
        diag[skin_idx, 1] += 50
        diag_uint8 = np.clip(diag, 0, 255).astype(np.uint8)
        for s in analysis.vector_shapes:
            sx, sy, sw, sh = s.bbox
            cv2.rectangle(diag_uint8, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
        return diag_uint8

    # ------------------------------------------------------------------
    # Raster upscaling
    # ------------------------------------------------------------------
    def _fit_scale(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize engine output to the exact requested size (model native scale may differ)."""
        if img.shape[1] == target_w and img.shape[0] == target_h:
            return img
        interp = cv2.INTER_AREA if img.shape[1] > target_w else cv2.INTER_CUBIC
        return cv2.resize(img, (target_w, target_h), interpolation=interp)

    def _upscale_raster(self, img_bgr: np.ndarray, scale: int, fast: bool) -> np.ndarray:
        self._init_engine()
        assert self.engine is not None
        if self.tta != (not fast):
            self.tta = not fast
            self.engine = None  # rebuild with the requested TTA setting
            self._init_engine()
        out = self.engine.upscale(img_bgr)
        H, W = img_bgr.shape[:2]
        return self._fit_scale(out, W * scale, H * scale)

    def _upscale_fidelity(self, img_bgr: np.ndarray, scale: int) -> np.ndarray:
        if not _TORCH_AVAILABLE or not os.path.exists(self.checkpoint_path):
            print("  [SmartUpscaler] fidelity model unavailable; using engine path")
            return self._upscale_raster(img_bgr, scale, fast=True)
        self._init_fidelity()
        assert self._fidelity_fn is not None
        out = tiled_run(self._fidelity_fn, img_bgr, 4, tile=self.tile or 256, pad=32)
        H, W = img_bgr.shape[:2]
        return self._fit_scale(out, W * scale, H * scale)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _upscale_document(self, img_bgr: np.ndarray, scale: int) -> np.ndarray:
        """Document path: classical restore (+ OCR classification if available).

        Returns the cleaned display page. Falls back to restore-only when no OCR
        backend is installed (restore still works offline).
        """
        try:
            from document_pipeline import run_document_pipeline
            res = run_document_pipeline(img_bgr, deskew=False, scale=scale)
            print(f"  [SmartUpscaler] document mode: {res.status_line}")
            return res.display_bgr
        except Exception as e:  # noqa: BLE001
            print(f"  [SmartUpscaler] document pipeline unavailable ({e}); restore-only")
            from document_restore import restore_document
            return restore_document(img_bgr, scale=scale)["display_bgr"]

    def upscale(self, img: np.ndarray, scale: int = 4, mode: str = "auto",
                fast: bool = True, grain_strength: float = 0.0,
                export_svg_path: Optional[str] = None,
                export_mask_path: Optional[str] = None) -> np.ndarray:
        t0 = time.time()
        is_uint8 = (img.dtype == np.uint8)
        img_uint8 = img if is_uint8 else (np.clip(img, 0, 1) * 255.0).round().astype(np.uint8)
        is_mono = (img_uint8.ndim == 2) or (img_uint8.shape[2] == 1)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR) if is_mono else img_uint8

        H, W = img_bgr.shape[:2]
        out_H, out_W = H * scale, W * scale

        # Document is the product: explicit mode, or auto-detect a page.
        if mode == "document" or (mode == "auto" and is_document(img_bgr)):
            doc_out = self._upscale_document(img_bgr, scale)
            print(f"  [SmartUpscaler] document {H}x{W} -> {doc_out.shape[1]}x{doc_out.shape[0]} "
                  f"in {time.time() - t0:.2f}s")
            out_res = cv2.cvtColor(doc_out, cv2.COLOR_BGR2GRAY) if is_mono else doc_out
            if not is_uint8:
                return out_res.astype(np.float32) / 255.0
            return out_res

        analysis = self.analyze_content(img_bgr)

        if export_mask_path:
            diag_map = self.generate_diagnostic_map(img_bgr, analysis)
            cv2.imwrite(export_mask_path, diag_map)
            print(f"  [SmartUpscaler] semantic map -> {export_mask_path}")

        if export_svg_path and analysis.vector_shapes:
            export_svg(analysis.vector_shapes, W, H, export_svg_path, scale=1.0)
            print(f"  [SmartUpscaler] SVG -> {export_svg_path}")

        if mode == "fidelity":
            raster_hr = self._upscale_fidelity(img_bgr, scale)
        else:
            raster_hr = self._upscale_raster(img_bgr, scale, fast)

        should_blend_vector = (mode in ("auto", "vector")) and (len(analysis.vector_shapes) > 0)
        if should_blend_vector:
            vec_canvas = render_vector_shapes(analysis.vector_shapes, out_w=out_W,
                                              out_h=out_H, scale=float(scale),
                                              base_canvas=raster_hr)
            diff = np.abs(vec_canvas.astype(np.float32) - raster_hr.astype(np.float32)).max(axis=2)
            vec_presence = (diff > 4.0).astype(np.float32)
            feather_size = max(3, int(scale) | 1)
            alpha_map = cv2.GaussianBlur(vec_presence, (feather_size, feather_size), 0)[:, :, np.newaxis]
            alpha_map = np.clip(alpha_map * 1.15, 0.0, 1.0)
            blended = vec_canvas.astype(np.float32) * alpha_map + \
                raster_hr.astype(np.float32) * (1.0 - alpha_map)
            final_bgr = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            final_bgr = raster_hr

        if grain_strength and grain_strength > 0:
            final_bgr = self._apply_grain(final_bgr, strength=grain_strength)

        elapsed = time.time() - t0
        engine_name = self.engine.name if self.engine else "fidelity"
        vector_count = len(analysis.vector_shapes) if should_blend_vector else 0
        print(f"  [SmartUpscaler] {H}x{W} -> {out_H}x{out_W} in {elapsed:.2f}s | "
              f"engine={engine_name} mode={mode} vectors={vector_count}")

        out_res = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY) if is_mono else final_bgr
        if not is_uint8:
            return out_res.astype(np.float32) / 255.0
        return out_res

    def _apply_grain(self, img_bgr: np.ndarray, strength: float = 0.018) -> np.ndarray:
        """Optional luminance-conditioned micro-grain (off by default)."""
        if strength <= 0.0:
            return img_bgr
        img_f = img_bgr.astype(np.float32) / 255.0
        lum = 0.114 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.299 * img_f[:, :, 2]
        weight = np.clip(4.0 * lum * (1.0 - lum), 0.15, 1.0)[:, :, np.newaxis]
        noise = np.random.normal(0, strength, img_f.shape).astype(np.float32)
        out = np.clip(img_f + noise * weight, 0.0, 1.0)
        return (out * 255.0).round().astype(np.uint8)


_DEFAULT_UPSCALER: Optional[SmartUpscaler] = None


def smart_upscale(image: np.ndarray, scale: int = 4, mode: str = "auto",
                  fast: bool = True, grain_strength: float = 0.0,
                  export_svg_path: Optional[str] = None,
                  export_mask_path: Optional[str] = None,
                  model_spec: str = "auto") -> np.ndarray:
    """Module-level convenience API: numpy array in -> numpy array out."""
    global _DEFAULT_UPSCALER
    if _DEFAULT_UPSCALER is None or _DEFAULT_UPSCALER.model_spec != model_spec:
        _DEFAULT_UPSCALER = SmartUpscaler(model_spec=model_spec)
    return _DEFAULT_UPSCALER.upscale(
        img=image, scale=scale, mode=mode, fast=fast, grain_strength=grain_strength,
        export_svg_path=export_svg_path, export_mask_path=export_mask_path)
