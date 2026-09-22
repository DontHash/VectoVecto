"""
discriminator.py — PatchGAN discriminator (70x70 receptive field).

Adversarial loss pushes SR output towards the natural-image distribution
by training a discriminator to tell SR patches apart from HR patches.

Architecture follows the classic Isola et al. PatchGAN design:
  4-layer (dropblock-free), 70x70 receptive field. The discriminator maps
  each 70x70 input region to a real/fake logit. Convolutional output (NxN
  grid) rather than a single scalar — better at texture-realism.

Output is a (B, 1, H/16, W/16) logits map. Used with a BCE-style loss on
the GAN training step in train_deep_sr.py.
"""
import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_channels=3, num_feat=64):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # receptive fields accumulate to 70x70 over 4 conv blocks of stride 2
        self.model = nn.Sequential(
            block(in_channels, num_feat, norm=False),     # 64
            block(num_feat, num_feat * 2),               # 128
            block(num_feat * 2, num_feat * 4),           # 256
            block(num_feat * 4, num_feat * 8),           # 512
            nn.Conv2d(num_feat * 8, 1, 4, stride=1, padding=1, bias=True),  # 1
        )

        # Kaiming init (helps GAN stability early on)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


def gan_loss_generator(logits_fake, target_real_val=1.0):
    """
    Generator's adversarial loss: BCE logits of discriminator(SR) vs target label.
    target_real_val=1.0 means generator wants discriminator to label SR as real.
    """
    bce = nn.BCEWithLogitsLoss()
    target = torch.full_like(logits_fake, target_real_val)
    return bce(logits_fake, target)


def gan_loss_discriminator(logits_real, logits_fake):
    """Discriminator adversarial loss: BCE of D(real) -> 1 + D(fake) -> 0."""
    bce = nn.BCEWithLogitsLoss()
    target_real = torch.ones_like(logits_real)
    target_fake = torch.zeros_like(logits_fake)
    return 0.5 * (bce(logits_real, target_real) + bce(logits_fake, target_fake))


if __name__ == "__main__":
    net = PatchDiscriminator(in_channels=3, num_feat=64)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"PatchDiscriminator params: {n_params/1e6:.2f}M")
    x = torch.randn(2, 3, 128, 128)
    out = net(x)
    print(f"shape: {x.shape} -> {out.shape}")  # expect (2,1,8,8)
    assert out.shape[0] == 2 and out.shape[1] == 1
    print("shape OK")