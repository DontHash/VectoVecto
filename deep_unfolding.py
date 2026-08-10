import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
import cv2

# ==========================================
# 1. Lightweight AI Prior (Tiny Denoiser)
# ==========================================
class TinyDenoiser(nn.Module):
    """
    A very lightweight 5-layer CNN that acts as a learned proximal operator.
    It takes an image and a noise level, and predicts the high-frequency residual.
    """
    def __init__(self, in_channels=1, num_features=32, num_layers=5):
        super(TinyDenoiser, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            
        # Output layer
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Predicts the noise/residual to subtract
        residual = self.net(x)
        return x - residual

# ==========================================
# 2. Pure Math Operations (Data Fidelity)
# ==========================================
def create_gaussian_kernel(sigma=1.2):
    kernel_size = int(np.ceil(sigma * 3) * 2 + 1)
    k1d = cv2.getGaussianKernel(kernel_size, sigma)
    k2d = np.outer(k1d, k1d)
    return torch.from_numpy(k2d).float().unsqueeze(0).unsqueeze(0) # [1, 1, K, K]

def _kernel_to_batch(kernel, B, C):
    """
    Normalize kernel to (B*C, 1, K, K) for grouped conv (groups = B*C).
    Accepts: (K,K), (1,1,K,K) [shared], or (B,1,1,K,K) [per-sample].
    """
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)          # (1,1,K,K)
    if kernel.dim() == 4:
        # shared kernel (1,1,K,K) -> tile to (B,1,1,K,K)
        kernel = kernel.expand(B, 1, kernel.shape[-2], kernel.shape[-1]).unsqueeze(1)
    # kernel is now (B,1,1,K,K); tile across channels
    K = kernel.shape[-1]
    return kernel.expand(B, C, 1, K, K).reshape(B * C, 1, K, K)

def D_H_forward(x, kernel, scale=2):
    """
    Forward degradation: blur (H) then decimate (D). x is [B, C, H, W].
    kernel: (K,K), (1,1,K,K) [shared] or (B,1,1,K,K) [per-sample].
    """
    B, C, H, W = x.shape
    kper = _kernel_to_batch(kernel, B, C)          # (B*C, 1, K, K)
    xf = x.reshape(1, B * C, H, W)                 # groups=B*C: in=out=B*C
    blurred = F.conv2d(xf, kper, padding=kernel.shape[-1] // 2, groups=B * C)
    blurred = blurred.reshape(B, C, H, W)
    # D: Decimate
    return blurred[:, :, ::scale, ::scale]

def D_H_transpose(e, kernel, scale=2, out_size=None):
    """
    Transpose degradation (adjoint): zero-insert (D^T) then blur (H^T).
    e is [B, C, H/s, W/s]; kernel like D_H_forward.
    """
    B, C, H_e, W_e = e.shape
    if out_size is None:
        out_size = (H_e * scale, W_e * scale)

    # D^T: zero-insertion
    upsampled = torch.zeros((B, C, out_size[0], out_size[1]), device=e.device, dtype=e.dtype)
    upsampled[:, :, ::scale, ::scale] = e

    # H^T: convolution with flipped kernel, per sample/channel
    flipped = torch.flip(kernel, dims=[-2, -1])
    kper = _kernel_to_batch(flipped, B, C)         # (B*C, 1, K, K)
    up_f = upsampled.reshape(1, B * C, out_size[0], out_size[1])
    grad = F.conv2d(up_f, kper, padding=flipped.shape[-1] // 2, groups=B * C)
    return grad.reshape(B, C, out_size[0], out_size[1])

# ==========================================
# 3. Deep Unfolding (Proximal Gradient Descent)
# ==========================================
class DeepUnfoldingSR(nn.Module):
    """
    Proximal Gradient Unrolling of the SR inverse problem
        x* = arg min_x  0.5 * || D H x - y ||_2^2  +  R(x)
    via    x_{k+1} = Denoiser( x_k - alpha_k * grad_data(x_k) )
    where the Denoiser is the learned proximal operator of the prior R.

    Tier B: now supports DRUNet (with noise-level map input) and TinyDenoiser
    (back-compat). When the denoiser accepts a noise-level map, it is tiled to
    the HR grid and passed each iteration (DPIR-style).
    """

    def __init__(self, denoiser, iterations=5, scale=2, step_size=0.2):
        super(DeepUnfoldingSR, self).__init__()
        self.denoiser = denoiser
        self.iterations = iterations
        self.scale = scale
        # Trainable step sizes for gradient descent per iteration.
        # Init LOW (0.2): with an identity-initialized denoiser the data
        # fidelity step must not overshoot; alphas learn upward as training.
        self.alphas = nn.Parameter(torch.ones(iterations) * step_size)

    def _denoise(self, x, sigma_map):
        """Denoise dispatch: DRUNet takes (x, sigma_map); TinyDenoiser takes x."""
        try:
            return self.denoiser(x, sigma_map)
        except TypeError:
            # TinyDenoiser-style single-arg forward
            return self.denoiser(x)

    def forward(self, y, blur_kernel, init_x=None, sigma_noise=None):
        """
        y: Low-res input image (B, C, h, w) in [0,1].
        blur_kernel: degradation blur kernel (1,1,K,K) for D*H.
        init_x: Initial guess for HR image (default: bicubic upsample of y).
        sigma_noise: noise sigma scalar OR (B,) tensor. Tiled to a (B,1,H,W)
            noise-level map and fed to DRUNet each iteration. If None,
            defaults to a moderate sigma (0.05) which DPIR uses for blind SR.
        """
        B, C, h, w = y.shape
        out_size = (h * self.scale, w * self.scale)

        if init_x is None:
            x = F.interpolate(y, scale_factor=self.scale, mode='bicubic', align_corners=False)
        else:
            x = init_x

        # Build the noise-level map for DRUNet (default 0.05, DPIR blind SR)
        if sigma_noise is None:
            sigma_t = torch.tensor(0.05, device=y.device)
        else:
            sigma_t = sigma_noise if torch.is_tensor(sigma_noise) else torch.tensor(float(sigma_noise))
        # Tile to (B, 1, H, W) and clamp to [0,1] -- the SR noise level is
        # conventionally in [0,1] when pixels are in [0,1] (per DRUNet convention).
        if sigma_t.dim() == 0:
            sigma_t = sigma_t.view(1, 1, 1, 1).expand(B, 1, *out_size).contiguous()
        else:
            sigma_t = sigma_t.view(-1, 1, 1, 1).expand(-1, -1, *out_size).contiguous()
        sigma_t = sigma_t.clamp(0.0, 1.0)

        for i in range(self.iterations):
            residual = D_H_forward(x, blur_kernel, self.scale) - y
            grad = D_H_transpose(residual, blur_kernel, self.scale, out_size)
            # Clamp the learned step size: unbounded alphas let the data-fidelity
            # step overshoot and the unfolding diverges (observed L1 spikes to
            # 1000+ at epoch boundaries). Deep-unfolding literature caps these.
            alpha_i = self.alphas[i].clamp(0.0, 0.5)
            x_half = x - alpha_i * grad
            x_half = x_half.clamp(0.0, 1.0)
            x = self._denoise(x_half, sigma_t)

        return x

if __name__ == "__main__":
    print("Deep Unfolding (Algorithm Unrolling) Model Initialized.")

    from drunet import DRUNet

    dummy_lr = torch.randn(1, 3, 64, 64)
    kernel = create_gaussian_kernel(sigma=1.2)
    denoiser = DRUNet(in_channels=3, num_feat=64, num_blocks=20)

    model = DeepUnfoldingSR(denoiser, iterations=3, scale=2)
    out_hr = model(dummy_lr, kernel, sigma_noise=0.05)
    print(f"DRUNet path | LR Input: {dummy_lr.shape} -> HR Output: {out_hr.shape}")
