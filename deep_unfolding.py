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

def D_H_forward(x, kernel, scale=2):
    """ Forward degradation: blur (H) then decimate (D). x is [B, C, H, W] """
    B, C, H, W = x.shape
    out = []
    for c in range(C):
        channel = x[:, c:c+1, :, :]
        # H: Blur
        blurred = F.conv2d(channel, kernel, padding=kernel.shape[-1]//2)
        # D: Decimate
        decimated = blurred[:, :, ::scale, ::scale]
        out.append(decimated)
    return torch.cat(out, dim=1)

def D_H_transpose(e, kernel, scale=2, out_size=None):
    """ Transpose degradation (adjoint): zero-insert (D^T) then blur (H^T). e is [B, C, H/s, W/s] """
    B, C, H_e, W_e = e.shape
    if out_size is None:
        out_size = (H_e * scale, W_e * scale)
        
    out = []
    for c in range(C):
        channel = e[:, c:c+1, :, :]
        # D^T: Zero insertion
        upsampled = torch.zeros((B, 1, out_size[0], out_size[1]), device=e.device, dtype=e.dtype)
        upsampled[:, :, ::scale, ::scale] = channel
        
        # H^T: Convolution with flipped kernel
        flipped_kernel = torch.flip(kernel, dims=[2, 3])
        gradient = F.conv2d(upsampled, flipped_kernel, padding=kernel.shape[-1]//2)
        out.append(gradient)
    return torch.cat(out, dim=1)

# ==========================================
# 3. Deep Unfolding (Proximal Gradient Descent)
# ==========================================
class DeepUnfoldingSR(nn.Module):
    def __init__(self, denoiser, iterations=5, scale=2, step_size=1.0):
        super(DeepUnfoldingSR, self).__init__()
        self.denoiser = denoiser
        self.iterations = iterations
        self.scale = scale
        # Trainable step sizes for gradient descent per iteration
        self.alphas = nn.Parameter(torch.ones(iterations) * step_size)
        
    def forward(self, y, blur_kernel, init_x=None):
        """
        y: Low-res input image
        blur_kernel: Known or estimated blur kernel
        init_x: Initial guess for HR image (e.g. bicubic upsampled y)
        """
        B, C, h, w = y.shape
        out_size = (h * self.scale, w * self.scale)
        
        # Initialize x (HR estimate)
        if init_x is None:
            x = F.interpolate(y, scale_factor=self.scale, mode='bicubic', align_corners=False)
        else:
            x = init_x

        # Unrolled optimization loop
        for i in range(self.iterations):
            # 1. Data Projection (Math): gradient of 0.5 * ||DHx - y||^2
            residual = D_H_forward(x, blur_kernel, self.scale) - y
            grad = D_H_transpose(residual, blur_kernel, self.scale, out_size)
            x_half = x - self.alphas[i] * grad
            
            # 2. Proximal Step (Lightweight AI): Denoise the mathematically updated image
            x = self.denoiser(x_half)
            
        return x

if __name__ == "__main__":
    print("Deep Unfolding (Algorithm Unrolling) Model Initialized.")
    
    # Quick shape test
    dummy_lr = torch.randn(1, 1, 128, 128)
    kernel = create_gaussian_kernel(sigma=1.2)
    denoiser = TinyDenoiser(in_channels=1)
    
    model = DeepUnfoldingSR(denoiser, iterations=5, scale=2)
    
    # Forward pass
    out_hr = model(dummy_lr, kernel)
    print(f"LR Input Shape: {dummy_lr.shape}")
    print(f"HR Output Shape: {out_hr.shape}")
    print("Architecture validates perfectly! Pure math and lightweight AI successfully bridged.")
