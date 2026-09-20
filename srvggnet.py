"""
srvggnet.py — SRVGGNetCompact (Real-ESRGAN "general v3" compact architecture).

Used by weights/realesr-general-x4v3.pth (~1.2M params, fast on small GPUs).
Key names are the raw `body.N.*` scheme used by the official release.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _activation(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"unsupported activation: {act_type}")


class SRVGGNetCompact(nn.Module):
    """Pixel-shuffle VGG-style SR net: n conv layers + one upsampler."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), _activation(act_type, num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), _activation(act_type, num_feat)]
        body += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1),
                 nn.PixelShuffle(upscale)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        base = nn.functional.interpolate(x, scale_factor=self.upscale,
                                         mode="nearest")
        return out + base


def load_srvgg_compact(path: str, device="cpu", num_feat=64, num_conv=32,
                       upscale=4, act_type="prelu") -> SRVGGNetCompact:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "params" in sd:
        sd = sd["params"]
    model = SRVGGNetCompact(num_feat=num_feat, num_conv=num_conv,
                            upscale=upscale, act_type=act_type)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "weights", "realesr-general-x4v3.pth")
    m = load_srvgg_compact(p)
    n = sum(x.numel() for x in m.parameters())
    print(f"loaded SRVGGNetCompact: {n/1e6:.3f}M params")
    with torch.no_grad():
        y = m(torch.rand(1, 3, 32, 32))
    print("forward:", tuple(y.shape))
    assert y.shape[-1] == 128 and y.shape[-2] == 128
    print("srvggnet OK")
