"""
unet_discriminator.py — Real-ESRGAN U-Net discriminator with spectral norm,
plus the relativistic average GAN (RaGAN) loss used by Real-ESRGAN.

Why not the old PatchGAN: the U-Net D enforces both local texture realism and
global structure consistency, which measurably reduces the "clay + isolated
glossy patches" failure mode of small PatchGAN training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        norm = nn.utils.spectral_norm
        self.skip_connection = skip_connection

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        return self.conv9(out)


class RaGANLoss(nn.Module):
    """Relativistic average GAN loss (Jolicoeur-Martineau), basicsr-compatible."""

    def __init__(self, real_label_val: float = 1.0, fake_label_val: float = 0.0):
        super().__init__()
        self.cri = nn.BCEWithLogitsLoss()
        self.cri_avg = nn.L1Loss()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

    def _target(self, x: torch.Tensor, is_real: bool) -> torch.Tensor:
        val = self.real_label_val if is_real else self.fake_label_val
        return torch.empty_like(x).fill_(val)

    def disc_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        real_t = self._target(real_pred, True)
        fake_t = self._target(fake_pred, False)
        loss_real = self.cri(real_pred - fake_pred.mean(), real_t)
        loss_fake = self.cri(fake_pred - real_pred.mean(), fake_t)
        return 0.5 * (loss_real + loss_fake)

    def gen_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        real_t = self._target(real_pred, True)   # generator wants fake to be "more real than real"
        fake_t = self._target(fake_pred, False)
        loss_real = self.cri(real_pred - fake_pred.mean(), fake_t)
        loss_fake = self.cri(fake_pred - real_pred.mean(), real_t)
        return 0.5 * (loss_real + loss_fake)


if __name__ == "__main__":
    d = UNetDiscriminatorSN(3, 64)
    n = sum(p.numel() for p in d.parameters())
    x = torch.randn(1, 3, 128, 128)
    y = d(x)
    print(f"UNetDiscriminatorSN: {n/1e6:.3f}M params, {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape[-2:] == (128, 128)
    loss_fn = RaGANLoss()
    real, fake = d(torch.randn(1, 3, 64, 64)), d(torch.randn(1, 3, 64, 64))
    dl, gl = loss_fn.disc_loss(real, fake), loss_fn.gen_loss(real.detach(), fake)
    print(f"RaGAN disc={dl.item():.4f} gen={gl.item():.4f}")
    print("ok")
