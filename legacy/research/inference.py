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

def apply_photographic_grain(img_np: np.ndarray, strength: float = 0.02) -> np.ndarray:
    """
    Applies subtle, luminance-conditioned organic film grain to break surface tension
    and restore natural skin micro-texture / photographic realism.
    Conditioned on the human contrast sensitivity curve (stronger in mid-tones, gentle in highlights/shadows).
    """
    if strength <= 0.0:
        return img_np

    img_f = img_np.astype(np.float32) / 255.0
    lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    # Parabolic mid-tone weighting: peak at lum=0.5, falls off near 0 and 1
    weight = 4.0 * lum * (1.0 - lum)
    weight = np.clip(weight, 0.15, 1.0)[:, :, np.newaxis]

    noise = np.random.normal(0, strength, img_f.shape).astype(np.float32)
    grain = noise * weight
    out = np.clip(img_f + grain, 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def save_image(tensor, path, mode='natural', grain_strength=0.015, use_tier_b=True):
    """
    Save image with natural photographic preservation or legacy mode.
    Tier-B DRUNet uses 'natural' mode to avoid cartoon/clay-like bilateral flattening.
    """
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)

    if mode == 'raw':
        # Pure neural reconstruction with zero post-processing
        Image.fromarray(tensor).save(path)
        return

    if mode == 'legacy' and not use_tier_b:
        # Legacy smoothing only for Tier-A ConvTranspose checkerboard
        tensor = cv2.bilateralFilter(tensor, d=5, sigmaColor=50, sigmaSpace=50)
        img = Image.fromarray(tensor)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3))
        img = ImageEnhance.Color(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.1)
        img.save(path)
        return

    # Natural Photographic Mode:
    # 1. Inject subtle organic micro-grain to break plastic surface tension
    if grain_strength > 0:
        tensor = apply_photographic_grain(tensor, strength=grain_strength)

    img = Image.fromarray(tensor)
    # 2. Gentle subpixel micro-contrast (radius 1, 20% - no haloing or posterization)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=20, threshold=2))
    img.save(path)



def predict_tta(model, x, kernel):
    """8-way Test-Time Augmentation (4 rotations + horizontal flips)."""
    B, C, H, W = x.shape
    scale = getattr(model, 'scale', 4)
    out = torch.zeros((B, C, H * scale, W * scale), device=x.device, dtype=x.dtype)
    for k in range(4):
        rot = torch.rot90(x, k, [2, 3])
        out += torch.rot90(model(rot, kernel), -k, [2, 3])
        flip = torch.flip(rot, [3])
        out += torch.rot90(torch.flip(model(flip, kernel), [3]), -k, [2, 3])
    return out / 8.0


def upscale_tiled(model, img_tensor, kernel, scale, tile_size=256, tile_pad=32, use_tta=True):
    """
    Seamless tiled super-resolution with overlapping cosine window blending.
    Guarantees low, constant VRAM usage regardless of input image size.
    """
    B, C, H, W = img_tensor.shape
    device = img_tensor.device

    # If image already fits in a single tile, process directly
    if H <= tile_size and W <= tile_size:
        print(f"Image fits in single tile ({H}x{W} <= {tile_size}). Running direct inference (TTA={use_tta})...")
        return predict_tta(model, img_tensor, kernel) if use_tta else model(img_tensor, kernel)

    stride = tile_size - 2 * tile_pad
    h_steps = max(1, int(np.ceil((H - 2 * tile_pad) / stride)))
    w_steps = max(1, int(np.ceil((W - 2 * tile_pad) / stride)))

    out_H, out_W = H * scale, W * scale
    out_tensor = torch.zeros((B, C, out_H, out_W), device=device, dtype=img_tensor.dtype)
    weights = torch.zeros((1, 1, out_H, out_W), device=device, dtype=img_tensor.dtype)

    # 2D Cosine window for seamless tile boundary blending
    patch_out_size = tile_size * scale
    wy = torch.sin(torch.linspace(0.01, float(np.pi - 0.01), patch_out_size, device=device))
    wx = torch.sin(torch.linspace(0.01, float(np.pi - 0.01), patch_out_size, device=device))
    tile_weight = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    total_tiles = h_steps * w_steps
    print(f"Tiled inference active: {H}x{W} -> {out_H}x{out_W} ({total_tiles} tiles, tile_size={tile_size}, pad={tile_pad}, TTA={use_tta})")

    tile_count = 0
    for i in range(h_steps):
        top = min(i * stride, max(0, H - tile_size))
        bottom = min(top + tile_size, H)
        if bottom - top < tile_size:
            top = max(0, bottom - tile_size)

        for j in range(w_steps):
            left = min(j * stride, max(0, W - tile_size))
            right = min(left + tile_size, W)
            if right - left < tile_size:
                left = max(0, right - tile_size)

            tile_count += 1
            patch = img_tensor[:, :, top:bottom, left:right]

            if use_tta:
                patch_out = predict_tta(model, patch, kernel)
            else:
                patch_out = model(patch, kernel)

            ptop, pbottom = top * scale, bottom * scale
            pleft, pright = left * scale, right * scale
            cur_weight = tile_weight[:, :, :pbottom - ptop, :pright - pleft]

            out_tensor[:, :, ptop:pbottom, pleft:pright] += patch_out * cur_weight
            weights[:, :, ptop:pbottom, pleft:pright] += cur_weight

    weights = torch.clamp(weights, min=1e-5)
    out_tensor = out_tensor / weights
    return out_tensor


# ==========================================
# USER CONFIGURATION PLACEHOLDER
# ==========================================
# You can paste the absolute path to your image here.
# If you leave this as is, a file dialog will pop up asking you to select an image visually!
IMAGE_PATH = "./idbhuntu.jpg"
OUTPUT_PATH = "./upscaled_result23.png"


def main():
    global IMAGE_PATH, OUTPUT_PATH
    
    # 1. Handle File Input (CLI arguments first, then configured path, then UI fallback)
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]

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
    # Tier B (deep_unfolding.py) loops per-channel and expects [1,1,K,K];
    # the legacy path's D_H_forward uses groups=C and expects [C,1,K,K].
    kernel_ch = 1 if use_tier_b else 3
    kernel = create_gaussian_kernel(sigma=1.2, channels=kernel_ch).to(device)

    # 5. Run Inference (Seamless Tiled with Overlap Blending)
    use_tta = "--no-tta" not in sys.argv
    tile_size = 256
    for arg in sys.argv:
        if arg.startswith("--tile-size="):
            try:
                tile_size = int(arg.split("=")[1])
            except ValueError:
                pass

    with torch.no_grad():
        out_tensor = upscale_tiled(
            model, img_tensor, kernel, sr_scale,
            tile_size=tile_size, tile_pad=32, use_tta=use_tta
        )

    # --- BUG FIX: Remove Padding ---
    if pad_h > 0 or pad_w > 0:
        out_tensor = out_tensor[:, :, :H*sr_scale, :W*sr_scale]

    # Phase 6: Optional Total Variation (TV) Minimization Refinement
    if "--tv-refine" in sys.argv:
        print("\nApplying Phase 6 Total Variation (TV / ROF) refinement...")
        try:
            from tv_refinement import tv_super_resolution_refine
            out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            lr_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if pad_h > 0 or pad_w > 0:
                lr_np = lr_np[:H, :W]
            refined_np = tv_super_resolution_refine(lr_np, out_np, scale=sr_scale, lambda_tv=0.005, num_iters=4)
            out_tensor = torch.from_numpy(refined_np).permute(2, 0, 1).unsqueeze(0).to(device)
            print("Phase 6 TV refinement completed successfully.")
        except Exception as e:
            print(f"Warning: TV refinement skipped due to: {e}")

    # Phase 4.5: Optional Hybrid Vector / Raster Decomposition
    if "--hybrid-vector" in sys.argv:
        print("\nApplying Phase 4.5 Hybrid Vector / Raster Decomposition...")
        try:
            from vector_raster_hybrid import hybrid_vector_raster_upscale
            out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            lr_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if pad_h > 0 or pad_w > 0:
                lr_np = lr_np[:H, :W]

            svg_out = None
            for arg in sys.argv:
                if arg.startswith("--svg="):
                    svg_out = arg.split("=")[1]

            def raster_cb(lr, scale):
                return (out_np * 255.0).round().astype(np.uint8)

            lr_uint8 = (lr_np * 255.0).round().astype(np.uint8)
            hybrid_np = hybrid_vector_raster_upscale(
                lr_uint8,
                scale=sr_scale,
                raster_engine=raster_cb,
                export_svg_path=svg_out
            )
            out_tensor = torch.from_numpy(hybrid_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            print("Phase 4.5 Hybrid Vector/Raster decomposition completed successfully.")
            if svg_out:
                print(f"Exported resolution-independent SVG to: {svg_out}")
        except Exception as e:
            print(f"Warning: Hybrid vector/raster skipped due to: {e}")

    # 6. Save and Finish (Natural Photographic Mode by default, bypassing clay-like bilateral filter)
    save_mode = 'natural'
    if "--legacy-filter" in sys.argv:
        save_mode = 'legacy'
    elif "--raw" in sys.argv:
        save_mode = 'raw'

    grain_val = 0.018 if use_tier_b else 0.0
    for arg in sys.argv:
        if arg.startswith("--grain="):
            try:
                grain_val = float(arg.split("=")[1])
            except ValueError:
                pass
        elif arg == "--no-grain":
            grain_val = 0.0

    save_image(out_tensor, OUTPUT_PATH, mode=save_mode, grain_strength=grain_val, use_tier_b=use_tier_b)
    new_size = (orig_size[0] * sr_scale, orig_size[1] * sr_scale)
    print(f"\nSUCCESS! Upscaled image saved to: {os.path.abspath(OUTPUT_PATH)}")
    print(f"New dimensions: {new_size[0]}x{new_size[1]} (mode={save_mode}, grain={grain_val})")
    print("Compare the results to see the texture preservation!")

if __name__ == "__main__":
    main()
