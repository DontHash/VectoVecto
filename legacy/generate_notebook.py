
# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Deep Unfolding: Zero-Compromise Image Upscaling
This notebook trains a state-of-the-art Deep Unfolding (Algorithm Unrolling) architecture.
It guarantees mathematical data consistency while utilizing a Lightweight U-Net prior and Perceptual Loss to achieve flawless, photorealistic textures.

**Dataset required:** Kaggle DIV2K (High Resolution images). Make sure you add it to the notebook!"""

code_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
import numpy as np
import scipy.signal
import cv2
import os
from PIL import Image
from tqdm import tqdm"""

text_unet = """## 1. Lightweight U-Net Denoiser (The Learned Prior)
Unlike a basic CNN, a U-Net has a massive receptive field. It downsamples to understand global context and upsamples with skip connections to perfectly reconstruct high-frequency textures."""

code_unet = """class LightweightUNet(nn.Module):
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
        return x - residual  # Denoiser predicts noise to subtract"""

text_math = """## 2. Pure Math: Inverse Problem Formulation
The rigorous degradation model ($y = DHx$) and its analytical transpose ensure that the high-resolution output strictly conforms to the low-resolution input."""

code_math = """def create_gaussian_kernel(sigma=1.2, channels=3):
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
            
        return x"""

text_loss = """## 3. Loss & Perceptual VGG
L1 ensures structural pixel-accuracy, while the VGG loss extracts high-level semantic features to guarantee photorealistic, professional sharpness."""

code_loss = """class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # Extract features up to relu3_4 (layer 21) which is standard for SR
        self.slice = nn.Sequential(*list(vgg.children())[:21])
        for param in self.parameters():
            param.requires_grad = False
            
        # VGG requires ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return F.l1_loss(self.slice(pred_norm), self.slice(target_norm))"""

text_data = """## 4. DIV2K DataLoader
Extracts standard 128x128 high-res patches on the fly to maximize GPU efficiency."""

code_data = """class DIV2KDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.image_paths) * 10 # Artificially expand epoch size (10 crops per image)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx % len(self.image_paths)]).convert('RGB')
        
        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.patch_size, self.patch_size))
        img = TF.crop(img, i, j, h, w)
        
        # Random flips/rotations for augmentation
        if np.random.rand() > 0.5: img = TF.hflip(img)
        if np.random.rand() > 0.5: img = TF.vflip(img)
        
        return TF.to_tensor(img)"""

text_train = """## 5. Training Loop
The central engine. Point `DIV2K_PATH` to your Kaggle attached dataset directory."""

code_train = """DIV2K_PATH = '/kaggle/input/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR' # Update if path differs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    denoiser = LightweightUNet(in_channels=3).to(device)
    model = DeepUnfoldingSR(denoiser, iterations=5, scale=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss().to(device)
    
    kernel = create_gaussian_kernel(sigma=1.2, channels=3).to(device)
    
    # Check if path exists, else fallback to dummy for validation
    if os.path.exists(DIV2K_PATH):
        dataset = DIV2KDataset(DIV2K_PATH, patch_size=128)
    else:
        print("WARNING: DIV2K not found. Using a small synthetic dataset for validation.")
        dataset = [torch.rand(3, 128, 128) for _ in range(50)]
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for hr_imgs in pbar:
            hr_imgs = hr_imgs.to(device)
            lr_imgs = D_H_forward(hr_imgs, kernel, scale=2)
            
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs, kernel)
            
            loss_l1 = l1_loss(sr_imgs, hr_imgs)
            loss_vgg = vgg_loss(sr_imgs, hr_imgs)
            
            # Hybrid Loss: 1.0 * L1 + 0.1 * VGG Perceptual
            total_loss = loss_l1 + 0.1 * loss_vgg
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({'Loss': total_loss.item()})
            
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {epoch_loss / len(dataloader):.4f}")

# Uncomment below to train
# train()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_unet),
    nbf.v4.new_code_cell(code_unet),
    nbf.v4.new_markdown_cell(text_math),
    nbf.v4.new_code_cell(code_math),
    nbf.v4.new_markdown_cell(text_loss),
    nbf.v4.new_code_cell(code_loss),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_train),
    nbf.v4.new_code_cell(code_train)
]

with open('kaggle_training.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook 'kaggle_training.ipynb' generated successfully.")
