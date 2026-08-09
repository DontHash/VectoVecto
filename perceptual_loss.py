"""
perceptual_loss.py — VGG19 perceptual (feature) loss for image restoration.

Used by train_deep_sr.py to push SR output towards perceptually natural
appearance (sharp edges, textures) instead of L1-only flatness.

Design:
  - Frozen pretrained VGG19 features (downloaded via torchvision on first run).
  - L1 distance between features of SR and HR at four relus:
      relu1_2 (low-level edges), relu2_2 (textures), relu3_3 (patterns),
      relu4_3 (high-level semantics).
  - Per-layer weights taken from the ESRGAN / Real-ESRGAN convention:
      relu1_2=0.1, relu2_2=0.1, relu3_3=1.0, relu4_3=1.0.
  - Input expected in [0,1] RGB; we normalize to ImageNet stats before VGG.

Note: images are x.encoder_features(); the loss returns a scalar.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class VGGPerceptualLoss(nn.Module):
    """
    Computes L1 distance between VGG19 features of two images.

    forward(sr, hr):
        sr, hr: (B, 3, H, W) in [0,1] RGB.
        returns: scalar perceptual loss (sum of weighted L1 feature distances).
    """

    def __init__(self, layer_weights=None, requires_grad=False):
        super().__init__()
        if layer_weights is None:
            # (block, conv_index, weight) — VGG19 features used in Real-ESRGAN
            layer_weights = {
                'relu1_2': 0.1, 'relu2_2': 0.1,
                'relu3_3': 1.0, 'relu4_3': 1.0,
            }
        self.layer_weights = layer_weights

        # ImageNet normalization stats baked into the module as buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        if not requires_grad:
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
        self.vgg = vgg

        # Map configured relu names to indices in the VGG19 feature module.
        # VGG19 layers (0-indexed): odd=conv, next=relu. relu1_2 = layer 3.
        self.layer_indices = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 17, 'relu4_3': 26,
        }

        # Sanity: all requested layers must be in the index map
        for name in self.layer_weights.keys():
            assert name in self.layer_indices, f"Unknown VGG layer: {name}"

    def _extract_features(self, x):
        """Run VGG up to each requested layer, returning dict name -> feature."""
        x = (x - self.mean) / self.std  # ImageNet normalize
        feats = {}
        max_idx = max(self.layer_indices.values())
        for i in range(max_idx + 1):
            x = self.vgg[i](x)
            for name, idx in self.layer_indices.items():
                if i == idx:
                    feats[name] = x
        return feats

    def forward(self, sr, hr):
        if not sr.requires_grad:
            # If SR doesn't need grad (e.g. validation), skip grad build
            with torch.no_grad():
                return self._forward_impl(sr, hr)
        return self._forward_impl(sr, hr)

    def _forward_impl(self, sr, hr):
        # SR keeps gradients; HR is target only
        f_sr = self._extract_features(sr)
        with torch.no_grad():
            f_hr = self._extract_features(hr)
        loss = 0.0
        for name, w in self.layer_weights.items():
            loss = loss + w * torch.nn.functional.l1_loss(f_sr[name], f_hr[name])
        return loss


if __name__ == "__main__":
    # Quick shape + backprop check (downloads ~80MB VGG on first run)
    print("Constructing VGGPerceptualLoss (torchvision will download VGG19)...")
    vgg_loss = VGGPerceptualLoss()
    sr = torch.rand(2, 3, 96, 96, requires_grad=True)
    hr = torch.rand(2, 3, 96, 96)
    loss = vgg_loss(sr, hr)
    print(f"loss scalar = {loss.item():.4f}")
    loss.backward()
    print("backward OK; grad on sr:", sr.grad is not None and sr.grad.abs().mean().item() > 0)