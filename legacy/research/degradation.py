"""
degradation.py — Real-ESRGAN second-order degradation pipeline.

Builds the (HR -> LR) degradation used to train SR models on REAL-looking
low-quality images, instead of the naive `D*H` (Gaussian blur + decimate)
that the old train.py used. This is what makes the trained model
simultaneously denoise + de-JPEG + upscale = "cleaned".

Pipeline per image (Real-ESRGAN "second-order" degradation, simplified):
  order 1: blur -> down -> noise -> JPEG
  order 2: blur -> down -> noise -> JPEG  (applied to the order-1 output)
  final:   sinc filter (simulates the AA of the capture pipeline)

  HR image -------------------> blur1 -> down1 -> noise1 -> JPEG1 (order1 output)
                                  -> blur2 -> down2 -> noise2 -> JPEG2 -> sinc -> LR

Outputs are (LR, kernel, sigma) where sigma is the std of the final noise
(varies per image and is passed to DRUNet as its noise-level-map).

Reference: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data", ICCV 2021 W, https://arxiv.org/abs/2107.10833
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from io import BytesIO


# Real-ESRGAN default blur kernel size (the canonical cfg uses 21x21 blur)
DEFAULT_BLUR_SIGMA = [0.2, 3.0]
DEFAULT_BLUR_KERNEL_SIZE = 21

# isotropic and anisotropic blur sigma ranges (mimics Real-ESRGAN)
DEFAULT_SIGMA_RANGE = [0.2, 3.0]

# noise sigma range in [0,1] units (image is float32 in [0,1])
DEFAULT_NOISE_SIGMA_RANGE = [0.0, 25.0 / 255.0]

# JPEG quality range — second pass uses more aggressive compression
DEFAULT_JPEG_RANGE_ORDER1 = [60, 95]
DEFAULT_JPEG_RANGE_ORDER2 = [30, 95]

# downscale methods to sample from
DOWNSCALE_METHODS = ['bicubic', 'bilinear']  # cv2.INTER_CUBIC == bicubic, INTER_LINEAR


def _random_isotropic_gaussian_kernel(sigma, ksize=DEFAULT_BLUR_KERNEL_SIZE):
    """Return ksize x ksize torch tensor for an isotropic gaussian blur."""
    if isinstance(sigma, (tuple, list)):
        sigma = random.uniform(*sigma)
    # 1D gaussian centered
    coords = torch.arange(ksize, dtype=torch.float32).sub_(ksize // 2)
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = torch.outer(gauss_1d, gauss_1d)
    return kernel


def _apply_blur(img_tensor, kernel):
    """Apply blur using conv2d (single kernel across channels). img: (B,C,H,W)."""
    B, C, H, W = img_tensor.shape
    pad = kernel.shape[0] // 2
    kernel = kernel.to(img_tensor).view(1, 1, kernel.shape[0], kernel.shape[0]).repeat(C, 1, 1, 1)
    out = F.conv2d(img_tensor, kernel, padding=pad, groups=C)
    return out


def _downscale(img_tensor, down_factor, mode='bicubic'):
    """Shrink image. down_factor > 1 means SHRINK (2 = half size).
    img (B,C,H,W) float. We translate to F.interpolate's scale_factor = 1/down_factor.
    """
    intp_scale = 1.0 / float(down_factor)
    if mode == 'bicubic':
        return F.interpolate(img_tensor, scale_factor=intp_scale, mode='bicubic',
                             align_corners=False, recompute_scale_factor=True)
    return F.interpolate(img_tensor, scale_factor=intp_scale, mode='bilinear',
                        align_corners=False, recompute_scale_factor=True)


def _add_noise(img_tensor, sigma):
    """Additive gaussian noise per-pixel. sigma is float, in [0,1] units. img is in [0,1]."""
    if sigma <= 0:
        return img_tensor
    noise = torch.randn_like(img_tensor) * sigma
    return (img_tensor + noise).clamp(0.0, 1.0)


def _jpeg_compress(img_tensor, quality_range):
    """JPEG compress a tensor by going through numpy + cv2.imencode.
    img_tensor: (1,C,H,W) in [0,1]. Returns same shape."""
    if img_tensor.shape[0] != 1:
        raise ValueError("JPEG step expects batch=1 (use per-sample)")
    q = random.randint(*quality_range)
    img_np = img_tensor[0].detach().cpu().numpy()  # numpy can't be on GPU
    img_np = np.transpose(img_np, (1, 2, 0))                              # HWC, RGB
    img_np = (img_np * 255.0).round().clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img_tensor
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)              # BGR uint8
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(img_tensor)


def _sinc_filter(img_tensor, kernel_size=21):
    """Sinc (ideal AA) filter to mimic camera anti-aliasing (Real-ESRGAN last step)."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    cutoff = 3.0  # modifiable; Real-ESRGAN uses 1.0 to ~8.0 occasionally
    X, Y = np.meshgrid(np.arange(kernel_size) - center, np.arange(kernel_size) - center)
    r = np.sqrt(X * X + Y * Y)
    np.divide(np.sin(cutoff * r), np.where(r == 0, 1.0, r), out=kernel, where=r != 0)
    kernel[r == 0] = 1.0
    kernel = kernel * (r <= cutoff)
    kernel = kernel / kernel.sum()
    k_t = torch.from_numpy(kernel).float().to(img_tensor)
    return _apply_blur(img_tensor, k_t)


def degrade_known_kernel(img, scale=4, blur_sigma_range=(0.7, 2.0),
                         noise_sigma_range=DEFAULT_NOISE_SIGMA_RANGE):
    """
    Self-consistent degradation for DEEP UNFOLDING training (USRNet/DPIR style).

    Generates LR with a SINGLE known Gaussian blur + decimation — exactly the
    degradation D_H_forward() assumes — plus additive noise and JPEG. The caller
    gets the ACTUAL kernel used, so the data-fidelity term in the unrolled
    optimizer is consistent with how y was made. Without this, the fixed
    sigma=1.2 kernel fights a random-blur LR and the unfolding diverges.

    Returns (lr, sigma_noise, kernel): lr (1,C,H/scale,W/scale), sigma (float),
    kernel (K,K) tensor that was used.
    """
    img = img.clone()
    sigma_blur = random.uniform(*blur_sigma_range)
    kernel = _random_isotropic_gaussian_kernel(sigma_blur, DEFAULT_BLUR_KERNEL_SIZE)
    img = _apply_blur(img, kernel.to(img))
    # decimate exactly like D_H_forward: sample [::scale, ::scale]
    img = img[:, :, ::scale, ::scale]

    sigma_noise = random.uniform(*noise_sigma_range)
    img = _add_noise(img, sigma_noise)
    img = _jpeg_compress(img, DEFAULT_JPEG_RANGE_ORDER2) if random.random() < 0.5 else img
    img = img.clamp(0.0, 1.0)
    return img, sigma_noise, kernel


def degrade_real_esrgan(img, scale=4):
    """
    Real-ESRGAN second-order degradation on a single float image tensor.

    Args
    ----
    img : (B=1, C=3, H, W) float tensor in [0,1] on any device.
          Must be B=1 (we process one image at a time for the JPEG step).
    scale : final downscale factor (4 for 4x SR training).

    Returns
    -------
    lr : (1, C, H_r, W_r) low-quality version of the image, approximately
         H/scale x W/scale. Yields a known degradation y = degrade(x).
    sigma_noise : final noise sigma passed to the denoiser's noise-level map.
    kernel : final blur kernel tensor (for the data-fidelity step in unfolding).
    """
    assert img.shape[0] == 1, "degrade_real_esrgan expects B=1"
    img = img.clone()

    # ===== order-1 =====
    sigma1 = random.uniform(*DEFAULT_SIGMA_RANGE)
    k1 = _random_isotropic_gaussian_kernel(sigma1)
    img = _apply_blur(img, k1.to(img))
    # downscale scale / order-1 factor (order-1 covers sqrt(scale)
    down1 = random.uniform(1.0, scale ** 0.5)
    img = _downscale(img, down1, mode=random.choice(DOWNSCALE_METHODS))
    n1 = random.uniform(*DEFAULT_NOISE_SIGMA_RANGE)
    img = _add_noise(img, n1)
    img = _jpeg_compress(img, DEFAULT_JPEG_RANGE_ORDER1)

    # ===== order-2 =====
    sigma2 = random.uniform(*DEFAULT_SIGMA_RANGE)
    k2 = _random_isotropic_gaussian_kernel(sigma2)
    img = _apply_blur(img, k2.to(img))
    # total downscale ratio must end up around `scale`
    down2 = max(1.0, scale / down1)
    img = _downscale(img, down2, mode=random.choice(DOWNSCALE_METHODS))
    n2 = random.uniform(*DEFAULT_NOISE_SIGMA_RANGE)
    img = _add_noise(img, n2)
    img = _jpeg_compress(img, DEFAULT_JPEG_RANGE_ORDER2)

    # ===== sinc final =====
    k_sinc = _random_isotropic_gaussian_kernel(random.uniform(*DEFAULT_SIGMA_RANGE))
    img = _sinc_filter(img) if random.random() < 0.5 else img  # optional

    img = img.clamp(0.0, 1.0)
    sigma_noise = max(n1, n2)  # report to noise-level map
    return img, sigma_noise, k_sinc


if __name__ == "__main__":
    # Smoke test: produce LR from HR
    print("Testing degrade_real_esrgan on a synthetic image...")
    import numpy as np
    np.random.seed(0)
    random.seed(0)
    H, W = 256, 256
    sample = torch.rand(1, 3, H, W)
    lr, sigma, kernel = degrade_real_esrgan(sample, scale=4)
    print(f"HR {sample.shape} -> LR {lr.shape}")
    print(f"final sigma_noise: {sigma:.4f}")
    print(f"kernel shape: {kernel.shape}")
    assert lr.shape[2] <= sample.shape[2] // 4 * 4
    print("degradation smoke OK")