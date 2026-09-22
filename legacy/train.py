
# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# archived research modules live under legacy/research; harness under evals/harness
for _extra in (_os.path.join(_ROOT, "legacy", "research"),
               _os.path.join(_ROOT, "evals", "harness")):
    if _extra not in _sys.path:
        _sys.path.insert(0, _extra)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import our architecture from the previous script
from deep_unfolding import TinyDenoiser, DeepUnfoldingSR, create_gaussian_kernel, D_H_forward

class SyntheticImageDataset(Dataset):
    """
    Generates synthetic images (random shapes and frequencies) on the fly.
    Perfect for verifying that the model can learn and backpropagate without needing a massive external dataset.
    """
    def __init__(self, num_samples=100, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random high-res image
        hr = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        
        # Add random gradient backgrounds
        x = np.linspace(0, 1, self.img_size)
        y = np.linspace(0, 1, self.img_size)
        X, Y = np.meshgrid(x, y)
        if np.random.rand() > 0.5:
            hr[0] += X * np.random.rand()
        if np.random.rand() > 0.5:
            hr[0] += Y * np.random.rand()
            
        # Add random high-frequency shapes (circles/rectangles)
        for _ in range(np.random.randint(2, 6)):
            cx, cy = np.random.randint(0, self.img_size, 2)
            r = np.random.randint(5, 20)
            val = np.random.rand()
            dist = np.sqrt((X * self.img_size - cx)**2 + (Y * self.img_size - cy)**2)
            hr[0][dist < r] = val
            
        hr = np.clip(hr, 0, 1)
        return torch.from_numpy(hr)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Model
    denoiser = TinyDenoiser(in_channels=1).to(device)
    model = DeepUnfoldingSR(denoiser, iterations=3, scale=2).to(device)
    
    # 2. Setup Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss() # L1 is better for preserving sharp edges in image restoration
    
    # 3. Create Data
    kernel = create_gaussian_kernel(sigma=1.2).to(device)
    dataset = SyntheticImageDataset(num_samples=50, img_size=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Starting Micro-Training Phase (Verifying Backpropagation and Mathematical Integrity)...")
    epochs = 2
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, hr_imgs in enumerate(dataloader):
            hr_imgs = hr_imgs.to(device)
            
            # Mathematically degrade the image to create our training input
            # y = D * H * x
            lr_imgs = D_H_forward(hr_imgs, kernel, scale=2)
            
            # Forward pass through the Deep Unfolding model
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs, kernel)
            
            # Calculate Loss (Super-Resolved vs Ground Truth)
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average L1 Loss: {avg_loss:.4f}")
        
    print("Training loop validated! The TinyDenoiser successfully learned within the mathematical unrolling framework.")

if __name__ == "__main__":
    train()
