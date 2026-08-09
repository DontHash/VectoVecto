import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from PIL import Image, ImageEnhance, ImageFilter
import tkinter as tk
from tkinter import filedialog

# Tier B: support DRUNet-based DeepUnfolding checkpoints
try:
    from drunet import DRUNet
    from deep_unfolding import DeepUnfoldingSR as DeepUnfoldingSRBase
    _TIER_B_AVAILABLE = True
except Exception:
    _TIER_B_AVAILABLE = False

# Default Tier-B checkpoint location (produced by train_deep_sr.py)
_TIER_B_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "artifacts", "deep_sr", "best_checkpoint.pth")
_TIER_B_SCALE = 4   # Tier B DRUNet model trains at 4x
_TIER_B_ITERS = 5    # unfolding K — match train_deep_sr default

# ==========================================
# 1. ARCHITECTURE (From Kaggle Notebook)
# ==========================================
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super(LightweightUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(features, features*2, 3, padding=1), nn.ReLU(inplace=True))
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features*2, features*4, 3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(features*4, features*2, 3, padding=1), nn.ReLU(inplace=True))
        
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(features*2, features, 3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(features, in_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        b = self.bottleneck(self.pool(e2))
        
        d2 = self.up2(b)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        residual = self.final(d1)
        return x - residual  # Denoiser predicts noise to subtract

def create_gaussian_kernel(sigma=1.2, channels=3):
    kernel_size = int(np.ceil(sigma * 3) * 2 + 1)
    k1d = cv2.getGaussianKernel(kernel_size, sigma)
    k2d = np.outer(k1d, k1d)
    k2d = torch.from_numpy(k2d).float().unsqueeze(0).unsqueeze(0)
    return k2d.repeat(channels, 1, 1, 1) # [C, 1, K, K]

def D_H_forward(x, kernel, scale=2):
    # Groups=C to apply the same kernel to all channels independently
    blurred = F.conv2d(x, kernel, padding=kernel.shape[-1]//2, groups=x.shape[1])
    return blurred[:, :, ::scale, ::scale]

def D_H_transpose(e, kernel, scale=2, out_size=None):
    B, C, H_e, W_e = e.shape
    if out_size is None:
        out_size = (H_e * scale, W_e * scale)
        
    upsampled = torch.zeros((B, C, out_size[0], out_size[1]), device=e.device, dtype=e.dtype)
    upsampled[:, :, ::scale, ::scale] = e
    
    flipped_kernel = torch.flip(kernel, dims=[2, 3])
    gradient = F.conv2d(upsampled, flipped_kernel, padding=kernel.shape[-1]//2, groups=C)
    return gradient

class DeepUnfoldingSR(nn.Module):
    def __init__(self, denoiser, iterations=5, scale=2, step_size=1.0):
        super(DeepUnfoldingSR, self).__init__()
        self.denoiser = denoiser
        self.iterations = iterations
        self.scale = scale
        self.alphas = nn.Parameter(torch.ones(iterations) * step_size)
        
    def forward(self, y, blur_kernel):
        B, C, h, w = y.shape
        out_size = (h * self.scale, w * self.scale)
        x = F.interpolate(y, scale_factor=self.scale, mode='bicubic', align_corners=False)

        for i in range(self.iterations):
            residual = D_H_forward(x, blur_kernel, self.scale) - y
            grad = D_H_transpose(residual, blur_kernel, self.scale, out_size)
            x_half = x - self.alphas[i] * grad
            x = self.denoiser(x_half)
            
        return x

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img.size

def save_image(tensor, path):
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    
    # --- Artifact Smoothing ---
    # The ConvTranspose2d upsampling layers can sometimes cause "checkerboard" block artifacts.
    # We apply a Bilateral Filter, which acts as a "smart blur" that smooths out blocky pixels 
    # and flat areas while keeping the sharp edges perfectly intact.
    import cv2
    tensor = cv2.bilateralFilter(tensor, d=5, sigmaColor=50, sigmaSpace=50)

    img = Image.fromarray(tensor)
    
    # --- Quality Enhancement (Math & Lightweight filtering) ---
    # 1. Sharpening (Unsharp Mask: math-based edge enhancement)
    # Reduced percent from 150 to 100 so it doesn't over-sharpen and bring back blocks.
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3))
    
    # 2. Vibrance (Color Saturation)
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(1.2) # 20% increase in color vibrance
    
    # 3. Contrast (Makes shadows darker and highlights brighter)
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(1.1) # 10% increase in contrast
    
    img.save(path)


# ==========================================
# USER CONFIGURATION PLACEHOLDER
# ==========================================
# You can paste the absolute path to your image here.
# If you leave this as is, a file dialog will pop up asking you to select an image visually!
IMAGE_PATH = "./idbhuntu.jpg"
OUTPUT_PATH = "./upscaled_result23.png"


def main():
    global IMAGE_PATH
    
    # 1. Handle File Input (Terminal + UI Fallback)
    if not os.path.exists(IMAGE_PATH):
        print("Opening file dialog to select an image...")
        try:
            root = tk.Tk()
            root.withdraw() # Hide the main tk window
            root.attributes("-topmost", True) # Bring dialog to front
            IMAGE_PATH = filedialog.askopenfilename(
                title="Select an image to upscale (2x)", 
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
            )
            root.destroy()
        except Exception as e:
            print(f"Could not open GUI file dialog. Please manually type the path into the IMAGE_PATH variable. Error: {e}")
            return
            
        if not IMAGE_PATH:
            print("No file selected. Exiting.")
            return

    # 2. Setup Device and Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("Initializing Deep Unfolding Math+AI Model...")
    # Tier B: prefer DRUNet checkpoint if it exists, else fall back to LightweightUNet
    use_tier_b = _TIER_B_AVAILABLE and os.path.exists(_TIER_B_CHECKPOINT)
    if use_tier_b:
        print(f"Using Tier-B DRUNet checkpoint: {_TIER_B_CHECKPOINT}")
        denoiser = DRUNet(in_channels=3, num_feat=64, num_blocks=20).to(device)
        model = DeepUnfoldingSRBase(denoiser, iterations=_TIER_B_ITERS, scale=_TIER_B_SCALE).to(device)
        ckpt = torch.load(_TIER_B_CHECKPOINT, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        model.eval()
        # scale used by TTA + downstream math
        sr_scale = _TIER_B_SCALE
    else:
        denoiser = LightweightUNet(in_channels=3).to(device)
        model = DeepUnfoldingSR(denoiser, iterations=5, scale=2).to(device)
        # 3. Load Weights (Tier A/legacy path)
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts', 'best_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading trained DIV2K weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print("Run Tier B training: python train_deep_sr.py   (or download checkpoints into artifacts/deep_sr/)")
            return
        model.eval()
        sr_scale = 2
    
    # 4. Load Image
    print(f"\nReading image: {IMAGE_PATH}")
    img_tensor, orig_size = load_image(IMAGE_PATH)
    img_tensor = img_tensor.to(device)
    print(f"Original dimensions: {orig_size[0]}x{orig_size[1]}")

    # --- Dimension Padding ---
    # LightweightUNet needs H,W divisible by 4 (two maxpools). DRUNet is fully
    # convolutional with stride=1, so any size works; but for both paths we
    # pad to a multiple of `sr_scale` so successive F.interpolate(scale=sr_scale)
    # gives a clean output and de-pad restores the original aspect ratio.
    B, C, H, W = img_tensor.shape
    divisor = max(4, sr_scale)
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        import torch.nn.functional as F
        # Use reflection padding to avoid edge artifacts
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        print(f"Padded image internally to {W+pad_w}x{H+pad_h} to satisfy architecture (divisor={divisor}).")

    # Mathematical kernel for degradation constraint
    kernel = create_gaussian_kernel(sigma=1.2, channels=3).to(device)

    # 5. Run Inference
    print(f"\nUpscaling image by {sr_scale}x using 8-way Test-Time Augmentation (TTA)...")
    print("Running the model 8 times to drastically improve quality and destroy artifacts.")
    with torch.no_grad():
        # Initialize an empty tensor to accumulate the 8 predictions
        B, C, H, W = img_tensor.shape
        out_tensor = torch.zeros((B, C, H*sr_scale, W*sr_scale), device=device, dtype=img_tensor.dtype)

        for k in range(4):
            # 1. Standard Rotations
            rot_img = torch.rot90(img_tensor, k, [2, 3])
            out_rot = model(rot_img, kernel)
            out_tensor += torch.rot90(out_rot, -k, [2, 3])

            # 2. Flipped + Rotations
            flip_img = torch.flip(rot_img, [3]) # horizontal flip
            out_flip = model(flip_img, kernel)
            out_tensor += torch.rot90(torch.flip(out_flip, [3]), -k, [2, 3])

        # Average the 8 passes
        out_tensor = out_tensor / 8.0

    # --- BUG FIX: Remove Padding ---
    if pad_h > 0 or pad_w > 0:
        out_tensor = out_tensor[:, :, :H*sr_scale, :W*sr_scale]

    # 6. Save and Finish
    save_image(out_tensor, OUTPUT_PATH)
    new_size = (orig_size[0] * sr_scale, orig_size[1] * sr_scale)
    print(f"\nSUCCESS! Upscaled image saved to: {os.path.abspath(OUTPUT_PATH)}")
    print(f"New dimensions: {new_size[0]}x{new_size[1]}")
    print("Compare the results to see the texture preservation!")

if __name__ == "__main__":
    main()
