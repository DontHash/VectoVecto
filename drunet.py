"""
drunet.py — DRUNet: a 20-layer ResNet denoiser with a noise-level-map input.

This is the learned proximal operator used by Deep Unfolding SR (DPIR/USRNet
lineage). It replaces the 5-layer TinyDenoiser in deep_unfolding.py:

  - 20 residual blocks of 64 conv channels (real image-prior capacity, vs 5).
  - Input channels = num_in_ch + 1: the image is concatenated with a
    noise-level map (constant sigma per image -> HxW tensor of sigma), so
    the SAME network handles blind SR across a range of noise levels.
  - NO BatchNorm (BN running stats depend on training distribution and break
    at inference on disjoint data). GroupNorm-free too — plain convs, the
    standard DRUNet recipe.
  - Residual learning: net predicts the noise residual; output = input - residual.

Refs:
  - Zhang et al., "Deep Plug-and-Play Image Restoration" (DPIR), 2021.
  - Zhang et al., "DRUNet" architecture, https://github.com/cszn/DPIR
"""
import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch, bias=True):
    """Conv(3x3) -> BN-less -> ReLU. No BN: keeps inference stable across dists."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x


class DRUNet(nn.Module):
    """
    Inputs : x_image (B, C, H, W) in [0,1]  AND  noise_level (B, 1, H, W)
    Output : denoised image (B, C, H, W) in ~[0,1]

    The noise_level map is tiled to (B,1,H,W) before forward() if a scalar is
    passed; see DRUNet.forward signature.
    """

    def __init__(self, in_channels=3, num_feat=64, num_blocks=20):
        super().__init__()
        self.in_channels = in_channels
        # +1 for the noise-level map channel
        self.head = nn.Conv2d(in_channels + 1, num_feat, 3, padding=1, bias=True)
        self.body = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(num_feat, in_channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Kaiming init for stable training start
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, noise_level_map):
        """
        x: (B, C, H, W) image in [0,1]
        noise_level_map: (B, 1, H, W) noise sigma per pixel in [0,1] (or scalar).
        """
        if noise_level_map.dim() <= 1:
            # scalar (dim 0) or per-batch (dim 1, shape (B,)) -> tile to (B,1,H,W)
            if noise_level_map.dim() == 0:
                noise_level_map = noise_level_map.view(1, 1, 1, 1).expand(x.shape[0], 1, *x.shape[2:]).contiguous()
            else:  # (B,)
                noise_level_map = noise_level_map.view(-1, 1, 1, 1).expand(-1, -1, *x.shape[2:]).contiguous()

        inp = torch.cat([x, noise_level_map], dim=1)
        f = self.relu(self.head(inp))
        f = self.body(f)
        residual = self.tail(f)
        return x - residual


if __name__ == "__main__":
    import numpy as np
    # Shape test
    net = DRUNet(in_channels=3, num_feat=64, num_blocks=20)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DRUNet params: {n_params/1e6:.2f}M")
    img = torch.randn(2, 3, 64, 64)
    sigma = torch.tensor([0.05, 0.1])
    out = net(img, sigma)
    assert out.shape == img.shape, f"shape mismatch {out.shape}"
    print(f"shape OK: in {img.shape} -> out {out.shape}")