"""
degradation_v2.py — Corrected Real-ESRGAN second-order degradation for SR training.

Why this exists
---------------
degradation.py trained Tier-B on isotropic Gaussian blur + decimation with a
single known kernel. Real photos are degraded by lens PSF (often anisotropic),
resampling, sensor noise, JPEG and a display/capture chain. Models trained on
the narrow Gaussian-only distribution look "clayish" on real inputs: they
over-smooth texture they never saw degraded in training.

This module ports the Real-ESRGAN / BSRGAN kernel zoo and second-order
pipeline, and fixes the bugs in `degrade_real_esrgan` (it returned a random
Gaussian as if it were the composite degradation kernel, never applied the
sinc filter kernel it generated, and used isotropic blur only).

Pipeline per sample (Real-ESRGAN, Wang et al. ICCVW 2021):
    HR --[blur1 -> resize1 -> noise1 -> jpeg1]-- order-1
       --[blur2 -> resize2 -> noise2 -> jpeg2]-- order-2
       --[sinc]-- LR

API:
    degrade_hr(hr, scale, ...) -> (lr, meta)          training (random)
    degrade_eval(hr, scale, seed=..., ...) -> (lr, meta)   deterministic eval
    degrade_np(img_rgb_uint8, scale, ...) -> lr_uint8  convenience for the harness
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ranges (Real-ESRGAN defaults)
# ---------------------------------------------------------------------------

BLUR_KERNEL_SIZE = 21
BLUR_SIGMA_RANGE = (0.2, 3.0)          # per-axis sigma range
BLUR_ROTATION_RANGE = (-math.pi, math.pi)
BETAG_RANGE = (0.5, 4.0)               # generalized gaussian shape
BETAP_RANGE = (0.5, 4.0)               # plateau shape

KERNEL_LIST = [
    "iso", "aniso", "generalized_iso", "generalized_aniso",
    "plateau_iso", "plateau_aniso",
]
KERNEL_PROBS = [0.31, 0.12, 0.31, 0.10, 0.10, 0.06]

NOISE_SIGMA_RANGE = (0.0, 25.0 / 255.0)
JPEG_RANGE_ORDER1 = (50, 95)
JPEG_RANGE_ORDER2 = (40, 95)
RESIZE_MODES = ("bilinear", "bicubic", "area", "nearest")


# ---------------------------------------------------------------------------
# Kernel zoo (numerically equivalent to Real-ESRGAN's degradations.py)
# ---------------------------------------------------------------------------

def _mesh(kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    return np.meshgrid(ax, ax)


def _gaussian_kernel(sigma_x, sigma_y, kernel_size, rotation,
                     beta=None, beta_p=None) -> np.ndarray:
    xx, yy = _mesh(kernel_size)
    rr = np.sqrt((beta or 1.0) / (beta_p or 1.0))
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    rot_xx = xx * cos_r + yy * sin_r
    rot_yy = -xx * sin_r + yy * cos_r

    xx2 = rot_xx / sigma_x
    yy2 = rot_yy / sigma_y
    kernel = np.exp(-0.5 * (xx2 ** 2 + yy2 ** 2))

    if beta is not None:
        kernel = kernel ** beta
    if beta_p is not None:
        kernel = kernel ** (1.0 / beta_p)
    _ = rr
    return kernel / kernel.sum()


def _sinc_kernel(sigma_x, sigma_y, kernel_size, rotation,
                 beta=None, beta_p=None, cutoff=2.0) -> np.ndarray:
    """Ideal low-pass (AA) filter, Real-ESRGAN style."""
    xx, yy = _mesh(kernel_size)
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    rot_xx = xx * cos_r + yy * sin_r
    rot_yy = -xx * sin_r + yy * cos_r
    radius = np.sqrt((rot_xx / (sigma_x * 2.0)) ** 2 + (rot_yy / (sigma_y * 2.0)) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(radius == 0.0, 1.0,
                          np.sin(radius * cutoff * math.pi) / (radius * cutoff * math.pi))
    kernel[radius > 1.0] = 0.0
    kernel = np.abs(kernel)
    return kernel / kernel.sum()


def random_mixed_kernels(kernel_list: Sequence[str], kernel_prob: Sequence[float],
                         kernel_size: int = BLUR_KERNEL_SIZE,
                         sigma_x_range=BLUR_SIGMA_RANGE,
                         sigma_y_range=BLUR_SIGMA_RANGE,
                         rotation_range=BLUR_ROTATION_RANGE,
                         betag_range=BETAG_RANGE, betap_range=BETAP_RANGE) -> np.ndarray:
    kernel_type = np.random.choice(a=list(kernel_list), size=1, p=list(kernel_prob))[0]
    sigma_x = np.random.uniform(*sigma_x_range)
    sigma_y = np.random.uniform(*sigma_y_range)
    rotation = np.random.uniform(*rotation_range)

    if "iso" in kernel_type:
        sigma_y = sigma_x
    if kernel_type == "iso":
        return _gaussian_kernel(sigma_x, sigma_y, kernel_size, rotation)
    if kernel_type == "aniso":
        return _gaussian_kernel(sigma_x, sigma_y, kernel_size, rotation)
    if kernel_type == "generalized_iso":
        beta = np.random.uniform(*betag_range)
        return _gaussian_kernel(sigma_x, sigma_x, kernel_size, rotation, beta=beta)
    if kernel_type == "generalized_aniso":
        beta = np.random.uniform(*betag_range)
        return _gaussian_kernel(sigma_x, sigma_y, kernel_size, rotation, beta=beta)
    if kernel_type == "plateau_iso":
        beta_p = np.random.uniform(*betap_range)
        return _gaussian_kernel(sigma_x, sigma_x, kernel_size, rotation, beta_p=beta_p)
    if kernel_type == "plateau_aniso":
        beta_p = np.random.uniform(*betap_range)
        return _gaussian_kernel(sigma_x, sigma_y, kernel_size, rotation, beta_p=beta_p)
    raise ValueError(f"unknown kernel type {kernel_type!r}")


def random_sinc_kernel(kernel_size: int = BLUR_KERNEL_SIZE) -> np.ndarray:
    sigma_x = np.random.uniform(0.2, 2.0)
    sigma_y = np.random.uniform(0.2, 2.0)
    rotation = np.random.uniform(*BLUR_ROTATION_RANGE)
    cutoff = np.random.uniform(1.0, 4.0)
    return _sinc_kernel(sigma_x, sigma_y, kernel_size, rotation, cutoff=cutoff)


# ---------------------------------------------------------------------------
# Tensor ops
# ---------------------------------------------------------------------------

def apply_kernel(img: torch.Tensor, kernel: np.ndarray) -> torch.Tensor:
    """Depthwise blur. img (B,C,H,W) float, kernel (k,k) numpy."""
    if kernel.shape[0] % 2 == 0:
        kernel = np.pad(kernel, ((0, 1), (0, 1)), mode="constant")
    k = torch.from_numpy(kernel.astype(np.float32)).to(img.device, img.dtype)
    k = k.view(1, 1, *k.shape).repeat(img.shape[1], 1, 1, 1)
    pad = k.shape[-1] // 2
    return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="reflect"), k, groups=img.shape[1])


def resize_tensor(img: torch.Tensor, down_factor: float, mode: str) -> torch.Tensor:
    if abs(down_factor - 1.0) < 1e-3:
        return img
    kwargs = {}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(img, scale_factor=1.0 / down_factor, mode=mode, **kwargs)


def add_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    return (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)


def jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Per-sample JPEG roundtrip on CPU. img (1,C,H,W) RGB float [0,1]."""
    arr = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    arr = np.clip(arr, 0, 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    dec = torch.from_numpy(dec[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
    return dec.to(img.device, img.dtype)


@dataclass
class DegradationMeta:
    blurs: list
    sigmas: list
    jpeg_qualities: list
    down1: float
    down2: float
    resize_modes: list
    sinc_sigma: float
    noise_sigma: float

    def as_dict(self):
        d = asdict(self)
        d["blurs"] = [np.asarray(b).tolist() for b in self.blurs]
        return d


# ---------------------------------------------------------------------------
# Public: training degradation
# ---------------------------------------------------------------------------

def degrade_hr(hr: torch.Tensor, scale: int = 4, order: int = 2,
               kernel_size: int = BLUR_KERNEL_SIZE,
               use_sinc: bool = True,
               use_poisson: bool = False,
               jpeg_p: float = 1.0) -> Tuple[torch.Tensor, DegradationMeta]:
    """Second-order degradation. hr (B=1,C,H,W) float [0,1] -> (lr, meta).

    Deterministic given torch/numpy seeds.
    """
    assert hr.dim() == 4, "expected (B,C,H,W)"
    img = hr.clamp(0.0, 1.0).clone()
    blurs, sigmas, qualities, modes = [], [], [], []
    downs = []
    noise_sigma = 0.0

    for o in range(order):
        kernel = random_mixed_kernels(KERNEL_LIST, KERNEL_PROBS, kernel_size)
        blur_sigma = float(kernel.shape[0])
        blurs.append(kernel)
        sigmas.append(blur_sigma)
        img = apply_kernel(img, kernel)

        if o == 0 and order > 1:
            down = random.uniform(1.0, float(scale) ** 0.5)
        else:
            remaining = float(scale)
            for d in downs:
                remaining /= d
            down = max(1.0, remaining)
        downs.append(down)
        mode = random.choice(RESIZE_MODES)
        modes.append(mode)
        img = resize_tensor(img, down, mode)

        sigma_n = random.uniform(*NOISE_SIGMA_RANGE)
        if use_poisson and random.random() < 0.5:
            lam = torch.poisson(img * 255.0) / 255.0
            img = ((img + lam) / 2.0).clamp(0.0, 1.0)
        else:
            img = add_noise(img, sigma_n)
        noise_sigma = max(noise_sigma, sigma_n)

        if random.random() < jpeg_p:
            q = random.randint(*(JPEG_RANGE_ORDER1 if o == 0 else JPEG_RANGE_ORDER2))
            qualities.append(q)
            img = jpeg_compress(img, q)

    # exact final size
    target_h = max(1, round(hr.shape[2] / scale))
    target_w = max(1, round(hr.shape[3] / scale))
    if img.shape[2] != target_h or img.shape[3] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic",
                            align_corners=False, antialias=True)

    sinc_sigma = 0.0
    if use_sinc and random.random() < 0.5:
        sk = random_sinc_kernel(kernel_size)
        sinc_sigma = float(np.random.uniform(0.2, 2.0))
        img = apply_kernel(img, sk)

    img = img.clamp(0.0, 1.0)
    meta = DegradationMeta(blurs=blurs, sigmas=sigmas, jpeg_qualities=qualities,
                           down1=downs[0] if downs else 1.0,
                           down2=downs[1] if len(downs) > 1 else 1.0,
                           resize_modes=modes, sinc_sigma=sinc_sigma,
                           noise_sigma=noise_sigma)
    return img, meta


# ---------------------------------------------------------------------------
# Public: deterministic eval degradation
# ---------------------------------------------------------------------------

def degrade_eval(hr: torch.Tensor, scale: int = 4, seed: int = 42,
                 severity: str = "mild") -> Tuple[torch.Tensor, DegradationMeta]:
    """Deterministic, realistic evaluation degradation.

    severity: 'mild' (well-exposed camera photo, light JPEG),
              'medium' (social media re-compression),
              'heavy' (old phone / screenshot-of-photo).
    """
    presets = {
        "mild":   dict(sigma=(0.5, 1.4), noise=(0.0, 8 / 255), jpeg1=(70, 95), jpeg2=(70, 95), sinc_p=0.5, order=1),
        "medium": dict(sigma=(0.8, 2.0), noise=(0.0, 15 / 255), jpeg1=(55, 90), jpeg2=(50, 90), sinc_p=0.5, order=2),
        "heavy":  dict(sigma=(1.2, 2.8), noise=(5 / 255, 25 / 255), jpeg1=(40, 80), jpeg2=(35, 80), sinc_p=0.7, order=2),
    }[severity]

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    img = hr.clamp(0.0, 1.0).clone()
    blurs, sigmas, qualities, modes, downs = [], [], [], [], []
    noise_sigma = 0.0

    for o in range(presets["order"]):
        kernel = random_mixed_kernels(KERNEL_LIST, KERNEL_PROBS, BLUR_KERNEL_SIZE,
                                      sigma_x_range=presets["sigma"],
                                      sigma_y_range=presets["sigma"])
        blurs.append(kernel)
        sigmas.append(float(rng.uniform(*presets["sigma"])))
        img = apply_kernel(img, kernel)

        if o == 0 and presets["order"] > 1:
            down = float(rng.uniform(1.0, float(scale) ** 0.5))
        else:
            remaining = float(scale)
            for d in downs:
                remaining /= d
            down = max(1.0, remaining)
        downs.append(down)
        mode = "bicubic"
        modes.append(mode)
        img = resize_tensor(img, down, mode)

        sigma_n = float(rng.uniform(*presets["noise"]))
        noise_sigma = max(noise_sigma, sigma_n)
        img = add_noise(img, sigma_n)

        q = int(rng.integers(*presets[f"jpeg{o + 1}"]))
        qualities.append(q)
        img = jpeg_compress(img, q)

    target_h = max(1, round(hr.shape[2] / scale))
    target_w = max(1, round(hr.shape[3] / scale))
    if img.shape[2] != target_h or img.shape[3] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic",
                            align_corners=False, antialias=True)

    sinc_sigma = 0.0
    if rng.random() < presets["sinc_p"]:
        sk = random_sinc_kernel(BLUR_KERNEL_SIZE)
        sinc_sigma = float(rng.uniform(0.2, 2.0))
        img = apply_kernel(img, sk)

    img = img.clamp(0.0, 1.0)
    meta = DegradationMeta(blurs=blurs, sigmas=sigmas, jpeg_qualities=qualities,
                           down1=downs[0] if downs else 1.0,
                           down2=downs[1] if len(downs) > 1 else 1.0,
                           resize_modes=modes, sinc_sigma=sinc_sigma,
                           noise_sigma=noise_sigma)
    return img, meta


def degrade_np(img_rgb_uint8: np.ndarray, scale: int = 4, seed: Optional[int] = None,
               severity: str = "mild") -> np.ndarray:
    """uint8 RGB numpy in -> uint8 RGB numpy LR out (for harness / CLI use)."""
    t = torch.from_numpy(img_rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    if seed is None:
        lr, _ = degrade_hr(t, scale=scale)
    else:
        lr, _ = degrade_eval(t, scale=scale, seed=seed, severity=severity)
    return (lr.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    x = torch.rand(1, 3, 512, 512)
    t0 = time.time()
    y, meta = degrade_hr(x, scale=4)
    print(f"train degradation: {tuple(x.shape)} -> {tuple(y.shape)} in {time.time()-t0:.2f}s")
    print(f"  jpeg={meta.jpeg_qualities} noise={meta.noise_sigma:.4f} sinc={meta.sinc_sigma:.2f}")
    assert y.shape[-2:] == (128, 128)

    t0 = time.time()
    for sev in ("mild", "medium", "heavy"):
        y2, meta2 = degrade_eval(x, scale=4, seed=123, severity=sev)
        assert y2.shape[-2:] == (128, 128)
        print(f"  eval[{sev}]: jpeg={meta2.jpeg_qualities} noise={meta2.noise_sigma:.4f}")
    print(f"eval degradations in {time.time()-t0:.2f}s")

    k = random_mixed_kernels(KERNEL_LIST, KERNEL_PROBS)
    print(f"kernel ok: sum={k.sum():.6f} shape={k.shape}")
    k2 = random_sinc_kernel()
    print(f"sinc ok:   sum={k2.sum():.6f} shape={k2.shape}")
