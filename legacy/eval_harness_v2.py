"""
eval_harness_v2.py — Professional evaluation harness for super-resolution.

Measures, on LR/HR pairs:
  * PSNR / SSIM            (pyiqa, RGB, [0,1])
  * LPIPS / DISTS          (pyiqa, perceptual)
  * Clay metrics           (HF-energy ratio, local-variance ratio,
                            composite clay score vs ground truth)

Pairs come from either:
  --hr-dir DIR     synthetic pairs: center-crop HR to --hr-size, then a
                   deterministic, realistic degradation (degradation_v2.degrade_eval)
  --lr-dir+--hr-dir  real pairs matched by filename stem

Models (adapters, comma-separated --models):
  bicubic | lanczos                       classical baselines
  x4plus                                  weights/RealESRGAN_x4plus.pth (RRDBNet)
  x4v3                                    weights/realesr-general-x4v3.pth (compact)
  tierb                                   artifacts/deep_sr/best_checkpoint.pth (unfolding)
  smart                                   SmartUpscaler full routing pipeline
  ncnn:<model-name>                       upscayl-bin.exe (e.g. ncnn:ultrasharp-4x)
  onnx:<path>                             ONNX Runtime (DirectML/CPU)

Examples:
  python eval_harness_v2.py --hr-dir data/DIV2K_valid_HR --images 20 --hr-size 512 \
      --models bicubic,lanczos,x4plus,x4v3,tierb --json out/metrics.json \
      --crops out/crops --csv out/metrics.csv
"""
from __future__ import annotations


# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import degradation_v2 as deg  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricSuite:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.metrics: Dict[str, Callable] = {}
        try:
            import pyiqa
            for name in ("psnr", "ssim", "lpips", "dists"):
                self.metrics[name] = pyiqa.create_metric(name, device=self.device)
            self.has_pyiqa = True
        except Exception as e:  # pragma: no cover
            print(f"[warn] pyiqa unavailable ({e}); falling back to cv2 PSNR/SSIM only")
            self.has_pyiqa = False

    @staticmethod
    def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    def score(self, pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.has_pyiqa:
            p = self._to_tensor(pred_bgr).to(self.device)
            r = self._to_tensor(ref_bgr).to(self.device)
            for name, fn in self.metrics.items():
                with torch.no_grad():
                    out[name] = float(fn(p, r).mean())
        else:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            out["psnr"] = float(peak_signal_noise_ratio(ref_bgr, pred_bgr, data_range=255))
            out["ssim"] = float(structural_similarity(ref_bgr, pred_bgr, channel_axis=2,
                                                      data_range=255))
        out.update(clay_metrics(pred_bgr, ref_bgr))
        return out


def _gray_f(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _hf_energy(g: np.ndarray) -> float:
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())


def _local_var(g: np.ndarray, k: int = 5) -> float:
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    return float(np.mean(np.maximum(m2 - m * m, 0.0)))


def _grad_energy(g: np.ndarray) -> float:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def clay_metrics(pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> Dict[str, float]:
    """Texture-fidelity metrics vs ground truth.

    hf_ratio / lv_ratio ~ 1.0 means the output carries the same texture energy
    as the reference. Ratios << 1 mean over-smoothing ("clay").
    clay_score in [0,1], 0 = no smoothing deficit (lower is better).
    """
    gp, gr = _gray_f(pred_bgr), _gray_f(ref_bgr)
    hf_r = _hf_energy(gp) / max(_hf_energy(gr), 1e-8)
    lv_r = _local_var(gp) / max(_local_var(gr), 1e-8)
    grad_r = _grad_energy(gp) / max(_grad_energy(gr), 1e-8)
    clay = 0.5 * max(0.0, 1.0 - min(hf_r, 1.0)) + 0.5 * max(0.0, 1.0 - min(lv_r, 1.0))
    return {
        "hf_ratio": hf_r,
        "lv_ratio": lv_r,
        "grad_ratio": grad_r,
        "clay_score": clay,
    }


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------

class Adapter:
    name: str = "base"
    scale: int = 4

    def __call__(self, lr_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def close(self):
        pass


class ClassicalAdapter(Adapter):
    def __init__(self, name: str, scale: int, interp: int):
        self.name = name
        self.scale = scale
        self.interp = interp

    def __call__(self, lr_bgr: np.ndarray) -> np.ndarray:
        h, w = lr_bgr.shape[:2]
        return cv2.resize(lr_bgr, (w * self.scale, h * self.scale),
                          interpolation=self.interp)


class TorchSRAdapter(Adapter):
    """Wraps an arbitrary torch SR model: (1,3,H,W) [0,1] -> (1,3,sH,sW)."""

    def __init__(self, name: str, model: torch.nn.Module, scale: int = 4,
                 device: str = "cuda", tile: int = 0, pad: int = 24,
                 tta: bool = False, fp16: bool = False):
        self.name = name
        self.model = model.eval().to(device)
        self.scale = scale
        self.device = torch.device(device)
        self.tile = tile
        self.pad = pad
        self.tta = tta
        self.fp16 = fp16 and self.device.type == "cuda"

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                return self.model(x).float()
        return self.model(x)

    def _forward_tta(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((1, x.shape[1], x.shape[2] * self.scale, x.shape[3] * self.scale),
                          device=x.device, dtype=x.dtype)
        for k in range(4):
            rot = torch.rot90(x, k, [2, 3])
            out += torch.rot90(self._forward(rot), -k, [2, 3])
            flip = torch.flip(rot, [3])
            out += torch.rot90(torch.flip(self._forward(flip), [3]), -k, [2, 3])
        return out / 8.0

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_tta(x) if self.tta else self._forward(x)

    def _tiled(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        s = self.scale
        tile, pad = self.tile, self.pad
        stride = tile - 2 * pad
        out = torch.zeros((1, 3, H * s, W * s), device=x.device, dtype=torch.float32)
        wsum = torch.zeros((1, 1, H * s, W * s), device=x.device, dtype=torch.float32)
        wy = torch.sin(torch.linspace(0.01, float(np.pi - 0.01), tile * s, device=x.device))
        wx = torch.sin(torch.linspace(0.01, float(np.pi - 0.01), tile * s, device=x.device))
        w2d = (wy.unsqueeze(1) * wx.unsqueeze(0)).view(1, 1, tile * s, tile * s)

        ys = list(range(0, max(H - 2 * pad, 1), stride)) or [0]
        xs = list(range(0, max(W - 2 * pad, 1), stride)) or [0]
        for y0 in ys:
            for x0 in xs:
                top = min(y0, max(0, H - tile)) if H > tile else 0
                left = min(x0, max(0, W - tile)) if W > tile else 0
                bottom = min(top + tile, H)
                right = min(left + tile, W)
                patch = x[:, :, top:bottom, left:right]
                ph, pw = patch.shape[-2:]
                pred = self._predict(patch)
                w_patch = w2d[:, :, :ph * s, :pw * s]
                out[:, :, top * s:bottom * s, left * s:right * s] += pred * w_patch
                wsum[:, :, top * s:bottom * s, left * s:right * s] += w_patch
        return out / torch.clamp(wsum, min=1e-5)

    def __call__(self, lr_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            try:
                if self.tile and (x.shape[-1] > self.tile or x.shape[-2] > self.tile):
                    y = self._tiled(x)
                elif self.tile:
                    y = self._tiled(x)
                else:
                    y = self._predict(x)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                tile = self.tile or 192
                save_tile, self.tile = self.tile, tile
                try:
                    y = self._tiled(x)
                finally:
                    self.tile = save_tile
        y = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return cv2.cvtColor((np.clip(y, 0, 1) * 255.0).round().astype(np.uint8),
                            cv2.COLOR_RGB2BGR)


class NCNNAdapter(Adapter):
    BIN = os.path.join(BASE_DIR, "upscayl-repo", "resources", "win", "bin", "upscayl-bin.exe")
    MODELS = os.path.join(BASE_DIR, "upscayl-repo", "resources", "models")

    def __init__(self, model_name: str, scale: int = 4):
        self.model_name = model_name
        self.name = f"ncnn:{model_name}"
        self.scale = scale
        if not os.path.exists(self.BIN):
            raise FileNotFoundError(self.BIN)

    def __call__(self, lr_bgr: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.png")
            out = os.path.join(td, "out.png")
            cv2.imwrite(inp, lr_bgr)
            cmd = [self.BIN, "-i", inp, "-o", out, "-m", self.MODELS,
                   "-n", self.model_name, "-s", str(self.scale), "-f", "png"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if not os.path.exists(out):
                raise RuntimeError(f"upscayl-bin failed: {r.stdout[-500:]} {r.stderr[-500:]}")
            return cv2.imread(out, cv2.IMREAD_COLOR)


class ONNXAdapter(Adapter):
    def __init__(self, path: str, scale: int = 4, device: str = "auto", tile: int = 0):
        import onnxruntime as ort
        self.name = f"onnx:{os.path.splitext(os.path.basename(path))[0]}"
        self.scale = scale
        self.tile = tile
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if device in ("auto", "dml") and "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider"] + providers
        elif device == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider"] + providers
        self.sess = ort.InferenceSession(path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]

    def __call__(self, lr_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb.transpose(2, 0, 1)[None]
        if self.tile:
            y = self._tiled(x)
        else:
            y = self._run(x)
        y = np.clip(y[0].transpose(1, 2, 0), 0, 1)
        return cv2.cvtColor((y * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _tiled(self, x: np.ndarray) -> np.ndarray:
        _, _, H, W = x.shape
        s, tile, pad = self.scale, self.tile, 16
        stride = max(tile - 2 * pad, 1)
        out = np.zeros((1, 3, H * s, W * s), dtype=np.float32)
        wsum = np.zeros((1, 1, H * s, W * s), dtype=np.float32)
        for y0 in range(0, max(H - 2 * pad, 1), stride):
            for x0 in range(0, max(W - 2 * pad, 1), stride):
                top = min(y0, max(0, H - tile))
                left = min(x0, max(0, W - tile))
                bottom, right = min(top + tile, H), min(left + tile, W)
                patch = x[:, :, top:bottom, left:right]
                pred = self._run(patch)
                ph, pw = pred.shape[-2], pred.shape[-1]
                out[:, :, top * s:top * s + ph, left * s:left * s + pw] += pred
                wsum[:, :, top * s:top * s + ph, left * s:left * s + pw] += 1.0
        return out / np.clip(wsum, 1e-5, None)


def build_adapter(spec: str, args) -> Adapter:
    spec = spec.strip()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if spec == "bicubic":
        return ClassicalAdapter("bicubic", args.scale, cv2.INTER_CUBIC)
    if spec == "lanczos":
        return ClassicalAdapter("lanczos", args.scale, cv2.INTER_LANCZOS4)

    if spec == "x4plus":
        from esrgan_inference import load_weights
        path = os.path.join(BASE_DIR, "weights", "RealESRGAN_x4plus.pth")
        model = load_weights(path, "cpu")
        return TorchSRAdapter("x4plus", model, args.scale, device,
                              tile=args.tile, tta=args.tta, fp16=args.fp16)

    if spec == "x4v3":
        from srvggnet import load_srvgg_compact
        path = os.path.join(BASE_DIR, "weights", "realesr-general-x4v3.pth")
        model = load_srvgg_compact(path, "cpu")
        return TorchSRAdapter("x4v3", model, args.scale, device,
                              tile=args.tile, tta=args.tta, fp16=args.fp16)

    if spec == "tierb":
        from drunet import DRUNet
        from deep_unfolding import DeepUnfoldingSR, create_gaussian_kernel
        ckpt_path = os.path.join(BASE_DIR, "artifacts", "deep_sr", "best_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        unfolding = DeepUnfoldingSR(
            DRUNet(in_channels=3, num_feat=64, num_blocks=20), iterations=5, scale=4)
        unfolding.load_state_dict(sd, strict=True)

        class TierB(torch.nn.Module):
            def __init__(self, u):
                super().__init__()
                self.u = u

            def forward(self, x):
                k = create_gaussian_kernel(sigma=1.2).to(x.device, x.dtype)
                return self.u(x, k)

        return TorchSRAdapter("tierb", TierB(unfolding), args.scale, device,
                              tile=0, tta=args.tta, fp16=False)

    if spec == "smart":
        from smart_upscaler import SmartUpscaler

        class SmartAdapter(Adapter):
            name = "smart"

            def __init__(self):
                self.engine = SmartUpscaler()
                self.scale = args.scale

            def __call__(self, lr_bgr):
                return self.engine.upscale(lr_bgr, scale=self.scale, mode="auto",
                                           fast=True, grain_strength=0.0)

        return SmartAdapter()

    if spec.startswith("ncnn:"):
        return NCNNAdapter(spec.split(":", 1)[1], args.scale)

    if spec.startswith("onnx:"):
        return ONNXAdapter(spec.split(":", 1)[1], args.scale, device, tile=args.tile)

    if spec.startswith("pth:"):
        spec = spec.split(":", 1)[1]
    if spec.endswith(".pth") and os.path.exists(spec):
        from sr_engine import build_generator
        model, arch = build_generator(spec, args.scale, "cpu")
        name = f"pth:{os.path.splitext(os.path.basename(spec))[0]}"
        return TorchSRAdapter(name, model, args.scale, device,
                              tile=args.tile, tta=args.tta, fp16=args.fp16)

    raise ValueError(f"unknown model spec: {spec!r}")


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def _list_images(d: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(d, "**", f"*{ext}"), recursive=True)
        files += glob.glob(os.path.join(d, "**", f"*{ext.upper()}"), recursive=True)
    return sorted(set(files))


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= size or w <= size:
        return img
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]


def build_pairs(args) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Returns [(name, lr_bgr, hr_bgr)]."""
    pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []

    if args.lr_dir and args.hr_dir:
        lr_files = _list_images(args.lr_dir)
        hr_index = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_images(args.hr_dir)}
        for lp in lr_files:
            stem = os.path.splitext(os.path.basename(lp))[0]
            hp = hr_index.get(stem)
            if hp is None:
                continue
            lr = cv2.imread(lp, cv2.IMREAD_COLOR)
            hr = cv2.imread(hp, cv2.IMREAD_COLOR)
            if lr is None or hr is None:
                continue
            pairs.append((stem, lr, hr))
            if args.images and len(pairs) >= args.images:
                break
        if not pairs:
            raise SystemExit("no matching LR/HR pairs found")
        return pairs

    if not args.hr_dir:
        raise SystemExit("provide --hr-dir (synthetic pairs) or --lr-dir with --hr-dir")

    hr_files = _list_images(args.hr_dir)
    if not hr_files:
        raise SystemExit(f"no images found in {args.hr_dir}")
    for i, hp in enumerate(hr_files):
        if args.images and i >= args.images:
            break
        hr_full = cv2.imread(hp, cv2.IMREAD_COLOR)
        if hr_full is None:
            continue
        hr = center_crop(hr_full, args.hr_size) if args.hr_size else hr_full
        t = torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        lr_t, _ = deg.degrade_eval(t, scale=args.scale, seed=args.seed + i,
                                   severity=args.severity)
        lr = cv2.cvtColor((lr_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0)
                          .round().clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        pairs.append((os.path.splitext(os.path.basename(hp))[0], lr, hr))
    if not pairs:
        raise SystemExit("failed to build any synthetic pairs")
    return pairs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_contact_crop(out_dir: str, name: str, model: str, lr: np.ndarray,
                      pred: np.ndarray, hr: np.ndarray, scale: int):
    os.makedirs(out_dir, exist_ok=True)
    h, w = hr.shape[:2]
    ch, cw = min(256, h), min(256, w)
    top, left = (h - ch) // 2, (w - cw) // 2
    base = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    panels = [base[top:top + ch, left:left + cw], pred[top:top + ch, left:left + cw],
              hr[top:top + ch, left:left + cw]]
    sep = np.full((ch, 4, 3), 255, dtype=np.uint8)
    row = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
    label = np.full((28, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(label, f"{name} | {model} | bicubic / model / HR", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f"{name}_{model.replace(':', '_')}.png"),
                np.vstack([label, row]))


def main():
    ap = argparse.ArgumentParser(description="SR evaluation harness v2")
    ap.add_argument("--hr-dir", default=None, help="HR images (synthetic pairs) or HR side of real pairs")
    ap.add_argument("--lr-dir", default=None, help="LR images for real pairs")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--images", type=int, default=0, help="max number of pairs (0=all)")
    ap.add_argument("--hr-size", type=int, default=512, help="center crop HR size (synthetic mode)")
    ap.add_argument("--severity", choices=["mild", "medium", "heavy"], default="mild")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="bicubic,lanczos,x4plus")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, default=0, help="tile size for torch models (0=whole image)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--crops", default=None, help="dir to save 3-panel crops")
    ap.add_argument("--crop-count", type=int, default=4, help="pairs to render crops for")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[harness] device={device} pairs-building...")
    pairs = build_pairs(args)
    print(f"[harness] {len(pairs)} pairs | models={args.models}")

    suite = MetricSuite(device)
    results: Dict[str, Dict[str, float]] = {}
    per_image: Dict[str, List[Dict[str, float]]] = {}
    timings: Dict[str, List[float]] = {}

    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        adapter = build_adapter(spec, args)
        name = adapter.name
        rows: List[Dict[str, float]] = []
        times: List[float] = []
        print(f"[harness] --- {name} ---")
        for idx, (img_name, lr, hr) in enumerate(pairs):
            t0 = time.time()
            pred = adapter(lr)
            dt = time.time() - t0
            times.append(dt)
            if pred.shape[:2] != hr.shape[:2]:
                pred = cv2.resize(pred, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            scores = suite.score(pred, hr)
            scores["image"] = img_name
            scores["seconds"] = dt
            rows.append(scores)
            if args.crops and idx < args.crop_count:
                save_contact_crop(args.crops, img_name, name, lr, pred, hr, args.scale)
            if (idx + 1) % 5 == 0:
                print(f"  [{idx+1}/{len(pairs)}] {img_name} "
                      f"psnr={scores.get('psnr', float('nan')):.2f} "
                      f"lpips={scores.get('lpips', float('nan')):.4f} "
                      f"clay={scores.get('clay_score', float('nan')):.4f} ({dt:.1f}s)")

        keys = [k for k in rows[0].keys() if k != "image"]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        results[name] = agg
        per_image[name] = rows
        timings[name] = times
        print(f"[harness] {name}: psnr={agg.get('psnr', float('nan')):.3f} "
              f"ssim={agg.get('ssim', float('nan')):.4f} "
              f"lpips={agg.get('lpips', float('nan')):.4f} "
              f"dists={agg.get('dists', float('nan')):.4f} "
              f"clay={agg.get('clay_score', float('nan')):.4f} "
              f"hf={agg.get('hf_ratio', float('nan')):.3f} "
              f"({np.mean(times):.1f}s/img)")
        adapter.close()

    order = sorted(results.items(), key=lambda kv: -(kv[1].get("psnr", 0)))
    print("\n=== SUMMARY (sorted by PSNR) ===")
    header = f"{'model':<22}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'DISTS':>8}{'clay':>7}{'hf':>6}{'s/img':>8}"
    print(header)
    print("-" * len(header))
    for name, m in order:
        print(f"{name:<22}{m.get('psnr', float('nan')):>8.3f}{m.get('ssim', float('nan')):>8.4f}"
              f"{m.get('lpips', float('nan')):>8.4f}{m.get('dists', float('nan')):>8.4f}"
              f"{m.get('clay_score', float('nan')):>7.4f}{m.get('hf_ratio', float('nan')):>6.3f}"
              f"{np.mean(timings[name]):>8.1f}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "scale": args.scale, "pairs": len(pairs),
                       "mean": results, "per_image": per_image}, f, indent=2)
        print(f"[harness] wrote {args.json}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "image", "psnr", "ssim", "lpips", "dists",
                        "hf_ratio", "lv_ratio", "grad_ratio", "clay_score", "seconds"])
            for name, rows in per_image.items():
                for r in rows:
                    w.writerow([name, r["image"]] + [f"{r.get(k, ''):.6f}" if isinstance(r.get(k), float)
                                                     else "" for k in
                                                     ["psnr", "ssim", "lpips", "dists", "hf_ratio",
                                                      "lv_ratio", "grad_ratio", "clay_score", "seconds"]])
        print(f"[harness] wrote {args.csv}")


if __name__ == "__main__":
    main()
