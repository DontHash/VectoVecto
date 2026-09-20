"""
sr_engine.py — unified production inference engine.

One entry point for every backend we ship:
    load_engine("auto" | "<path>.pth" | "<path>.onnx" | "ncnn:<name>" | "bicubic" | "lanczos")

* TorchEngine  — RRDBNet / SRVGGNetCompact / train_v2 checkpoints, tiled fp16 on CUDA
* ONNXEngine   — onnxruntime (DirectML on Windows, CUDA/CPU fallback), tiled
* NcnnEngine   — upscayl-bin.exe (Vulkan) with the bundled upscayl models
* ClassicalEngine — bicubic / Lanczos

All engines take/return BGR uint8 numpy arrays and expose `.scale` and `.name`.
Tiling uses cosine-shaded overlaps; no seams, bounded VRAM (fits a 4 GB RTX 2050).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Callable, Optional

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

DEFAULT_TIER_C = os.path.join(BASE_DIR, "artifacts", "tier_c", "g_ema.pth")
DEFAULT_ONNX_DIR = os.path.join(BASE_DIR, "weights", "onnx")
DEFAULT_X4PLUS = os.path.join(BASE_DIR, "weights", "RealESRGAN_x4plus.pth")
DEFAULT_X4V3 = os.path.join(BASE_DIR, "weights", "realesr-general-x4v3.pth")
NCNN_BIN = os.path.join(BASE_DIR, "upscayl-repo", "resources", "win", "bin", "upscayl-bin.exe")
NCNN_MODELS = os.path.join(BASE_DIR, "upscayl-repo", "resources", "models")


# ---------------------------------------------------------------------------
# Model detection / construction
# ---------------------------------------------------------------------------

def detect_arch(sd) -> str:
    keys = list(sd.keys())
    if any(k.startswith("conv_first") or k.startswith("model.0.") for k in keys):
        return "rrdbnet"
    if any(k.startswith("body.") and k.endswith(".weight") and sd[k].dim() == 4
           and sd[k].shape[0] == 64 and sd[k].shape[1] == 64 for k in keys):
        return "srvgg"
    if any(k.startswith("denoiser.") for k in keys):
        return "unfolding"
    raise ValueError(f"cannot detect arch from state dict keys: {keys[:4]}")


def load_state_dict_any(path: str):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict):
        for key in ("params_ema", "params", "g", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    return sd


def build_generator(path: str, scale: int = 4, device: str = "cpu"):
    """Returns (model, arch). Handles rrdbnet / srvgg checkpoints."""
    import torch

    sd = load_state_dict_any(path)
    arch = detect_arch(sd)
    if arch == "rrdbnet":
        from rrdbnet import RRDBNet
        model = RRDBNet(3, 3, 64, 23, 32, scale=scale)
    elif arch == "srvgg":
        from srvggnet import SRVGGNetCompact
        model = SRVGGNetCompact(3, 3, 64, 32, upscale=scale)
    else:
        raise ValueError("deep-unfolding checkpoints are not handled by sr_engine; "
                         "use SmartUpscaler's fidelity mode")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[sr_engine] load: {len(missing)} missing / {len(unexpected)} unexpected keys")
    model.eval().to(device)
    return model, arch


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Tiled inference (shared by torch/ONNX engines)
# ---------------------------------------------------------------------------

def _cosine_window(size: int) -> np.ndarray:
    w = np.sin(np.linspace(0.01, np.pi - 0.01, size, dtype=np.float32))
    return w


def tiled_run(fn: Callable[[np.ndarray], np.ndarray], bgr: np.ndarray, scale: int,
              tile: int, pad: int = 16) -> np.ndarray:
    """fn: BGR uint8 (h,w,3) -> BGR uint8 (s*h, s*w, 3). Overlapping cosine blend."""
    h, w = bgr.shape[:2]
    if tile <= 0 or (h <= tile and w <= tile):
        return fn(bgr)

    stride = max(tile - 2 * pad, 8)
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    wsum = np.zeros((h * scale, w * scale, 1), dtype=np.float32)

    ys = list(range(0, max(h - 2 * pad, 1), stride)) or [0]
    xs = list(range(0, max(w - 2 * pad, 1), stride)) or [0]
    for y0 in ys:
        for x0 in xs:
            top = min(y0, max(h - tile, 0))
            left = min(x0, max(w - tile, 0))
            bottom = min(top + tile, h)
            right = min(left + tile, w)
            patch = bgr[top:bottom, left:right]
            pred = fn(patch)
            ph, pw = pred.shape[:2]
            wy = _cosine_window(ph)[:, None]
            wx = _cosine_window(pw)[None, :]
            win = (wy * wx)[:, :, None]
            out[top * scale:top * scale + ph, left * scale:left * scale + pw] += pred.astype(np.float32) * win
            wsum[top * scale:top * scale + ph, left * scale:left * scale + pw] += win
    wsum = np.clip(wsum, 1e-5, None)
    return np.clip(out / wsum, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class Engine:
    name = "base"
    scale = 4
    arch = "?"

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} scale={self.scale} arch={self.arch}>"


class ClassicalEngine(Engine):
    def __init__(self, name: str = "bicubic", scale: int = 4):
        self.name = name
        self.scale = scale
        self.interp = cv2.INTER_CUBIC if name == "bicubic" else cv2.INTER_LANCZOS4
        self.arch = "classical"

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        return cv2.resize(bgr, (w * self.scale, h * self.scale),
                          interpolation=self.interp)


class TorchEngine(Engine):
    def __init__(self, path: str, scale: int = 4, device: str = "auto",
                 tile: int = 192, pad: int = 16, fp16: bool = True, tta: bool = False):
        import torch
        self.device = pick_device(device)
        self.model, self.arch = build_generator(path, scale, self.device)
        self.scale = scale
        self.tile = tile
        self.pad = pad
        self.fp16 = fp16 and self.device == "cuda"
        self.tta = tta
        self.name = f"torch:{os.path.splitext(os.path.basename(path))[0]}"
        self._torch = torch

    def _forward(self, x):
        torch = self._torch
        if self.fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                return self.model(x).float()
        return self.model(x)

    def _predict(self, x):
        torch = self._torch
        if not self.tta:
            return self._forward(x)
        out = torch.zeros((1, 3, x.shape[2] * self.scale, x.shape[3] * self.scale),
                          device=x.device, dtype=torch.float32)
        for k in range(4):
            rot = torch.rot90(x, k, [2, 3])
            out += torch.rot90(self._forward(rot), -k, [2, 3])
            flip = torch.flip(rot, [3])
            out += torch.rot90(torch.flip(self._forward(flip), [3]), -k, [2, 3])
        return out / 8.0

    def _fn(self, patch_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self._predict(x)
        y = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return cv2.cvtColor((np.clip(y, 0, 1) * 255).round().astype(np.uint8),
                            cv2.COLOR_RGB2BGR)

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        return tiled_run(self._fn, bgr, self.scale, self.tile, self.pad)


class ONNXEngine(Engine):
    def __init__(self, path: str, scale: int = 4, device: str = "auto",
                 tile: int = 192, pad: int = 16):
        import onnxruntime as ort
        self.path = path
        self.scale = scale
        self.tile = tile
        self.pad = pad
        self.name = f"onnx:{os.path.splitext(os.path.basename(path))[0]}"
        self.arch = "onnx"
        avail = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device in ("auto", "dml") and "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in avail and device != "cpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]

    def _fn(self, patch_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb.transpose(2, 0, 1)[None]
        y = np.clip(self._run(x)[0].transpose(1, 2, 0), 0, 1)
        return cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        return tiled_run(self._fn, bgr, self.scale, self.tile, self.pad)


class NcnnEngine(Engine):
    def __init__(self, model_name: str, scale: int = 4):
        self.model_name = model_name
        self.scale = scale
        self.arch = "ncnn"
        self.name = f"ncnn:{model_name}"
        if not os.path.exists(NCNN_BIN):
            raise FileNotFoundError(NCNN_BIN)

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.png")
            out = os.path.join(td, "out.png")
            cv2.imwrite(inp, bgr)
            cmd = [NCNN_BIN, "-i", inp, "-o", out, "-m", NCNN_MODELS,
                   "-n", self.model_name, "-s", str(self.scale), "-f", "png"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if not os.path.exists(out):
                raise RuntimeError(f"upscayl-bin failed: {r.stdout[-400:]} {r.stderr[-400:]}")
            return cv2.imread(out, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _tier_c_candidates():
    return [
        os.path.join(BASE_DIR, "artifacts", "tier_c", "g_ema.pth"),
        os.path.join(BASE_DIR, "artifacts", "tier_c", "latest.pth"),
        os.path.join(DEFAULT_ONNX_DIR, "tier_c_x4_fp16.onnx"),
        os.path.join(DEFAULT_ONNX_DIR, "tier_c_x4.onnx"),
    ]


def _x4plus_candidates():
    # NOTE: on the RTX 2050 DirectML is ~17x slower than PyTorch fp16 for RRDBNet,
    # so torch checkpoints come first; ONNX stays for CLI-free deployments.
    return [
        DEFAULT_X4PLUS,
        os.path.join(DEFAULT_ONNX_DIR, "x4plus_fp16.onnx"),
        os.path.join(DEFAULT_ONNX_DIR, "x4plus.onnx"),
    ]


def load_engine(spec: str = "auto", scale: int = 4, device: str = "auto",
                tile: Optional[int] = None, fp16: bool = True, tta: bool = False) -> Engine:
    """Resolve a model spec into a ready Engine.

    'auto' prefers: fine-tuned tier_c -> x4plus -> ncnn UltraSharp -> bicubic.
    """
    dev = pick_device(device)
    if tile is None:
        tile = 192 if dev == "cuda" else 256

    if spec in ("bicubic", "lanczos"):
        return ClassicalEngine(spec, scale)

    if spec == "x4plus":
        return TorchEngine(DEFAULT_X4PLUS, scale, dev, tile, fp16=fp16, tta=tta)
    if spec == "x4v3":
        return TorchEngine(DEFAULT_X4V3, scale, dev, tile, fp16=fp16, tta=tta)

    if spec == "auto":
        for cand in _tier_c_candidates():
            if os.path.exists(cand):
                print(f"[sr_engine] auto -> {cand}")
                if cand.endswith(".onnx"):
                    return ONNXEngine(cand, scale, dev, tile)
                return TorchEngine(cand, scale, dev, tile, fp16=fp16, tta=tta)
        for cand in _x4plus_candidates():
            if os.path.exists(cand):
                print(f"[sr_engine] auto -> {cand}")
                if cand.endswith(".onnx"):
                    return ONNXEngine(cand, scale, dev, tile)
                return TorchEngine(cand, scale, dev, tile, fp16=fp16, tta=tta)
        if os.path.exists(NCNN_BIN):
            print("[sr_engine] auto -> ncnn:ultrasharp-4x")
            return NcnnEngine("ultrasharp-4x", scale)
        print("[sr_engine] auto -> bicubic (no models found)")
        return ClassicalEngine("bicubic", scale)

    if spec.startswith("ncnn:"):
        return NcnnEngine(spec.split(":", 1)[1], scale)

    if spec.endswith(".onnx"):
        return ONNXEngine(spec, scale, dev, tile)

    if os.path.exists(spec):
        return TorchEngine(spec, scale, dev, tile, fp16=fp16, tta=tta)

    raise FileNotFoundError(f"model spec not found: {spec}")


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="sr_engine single-image CLI")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    eng = load_engine(args.model, args.scale, args.device, args.tile,
                      fp16=not args.no_fp16, tta=args.tta)
    print(eng)
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.input}")
    t0 = time.time()
    out = eng.upscale(img)
    dt = time.time() - t0
    cv2.imwrite(args.output, out)
    mp = (img.shape[0] * img.shape[1]) / 1e6
    print(f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]} in {dt:.2f}s "
          f"({dt / max(mp, 1e-6):.2f}s/MP)")
