
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
import cv2
import os
import numpy as np
from drunet import DRUNet
from deep_unfolding import DeepUnfoldingSR, create_gaussian_kernel
from inference import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = os.path.join('artifacts', 'deep_sr', 'best_checkpoint.pth')
denoiser = DRUNet(in_channels=3, num_feat=64, num_blocks=20).to(device)
model = DeepUnfoldingSR(denoiser, iterations=5, scale=4).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

img_t, orig_size = load_image('De1.jpg')
img_t = img_t.to(device)
kernel = create_gaussian_kernel(sigma=1.2).to(device)

with torch.no_grad():
    out = model(img_t, kernel)

raw = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
raw_uint8 = (raw * 255.0).round().astype(np.uint8)
# img_t in load_image was loaded via PIL convert('RGB'), so raw_uint8 is RGB
cv2.imwrite('De1_raw_drunet.png', cv2.cvtColor(raw_uint8, cv2.COLOR_RGB2BGR))
print('Saved De1_raw_drunet.png successfully.')
