"""
predict.py — run a trained deva_crnn checkpoint on line crops.

Usage:
    python -m deva_crnn.predict --ckpt out/deva_crnn/ckpt.pt \
        --images-dir data/doc_eval/heidata_lines/pages --out texts.tsv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from deva_crnn.charset import decode
from deva_crnn.data import normalize_line
from deva_crnn.model import CRNN


def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CRNN(n_classes=len(ckpt["charset"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, ckpt["charset"]


def recognize_lines(model, charset: List[str], images_bgr: List[np.ndarray],
                    device: str = "cpu") -> List[str]:
    if not images_bgr:
        return []
    batch = np.stack([normalize_line(im) for im in images_bgr]).astype(np.float32)
    x = torch.from_numpy(batch / 255.0).unsqueeze(1)
    x = ((x - 0.5) / 0.5).to(device)
    with torch.no_grad():
        preds = model(x).argmax(-1).permute(1, 0)
    return [decode(p.tolist(), charset) for p in preds]


def main():
    ap = argparse.ArgumentParser(description="deva_crnn line inference")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, charset = load_model(args.ckpt, args.device)
    names = sorted(f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    imgs = [cv2.imread(os.path.join(args.images_dir, n)) for n in names]
    keep = [(n, im) for n, im in zip(names, imgs) if im is not None]
    texts = recognize_lines(model, charset, [im for _, im in keep], args.device)
    with open(args.out, "w", encoding="utf-8") as f:
        for (n, _), t in zip(keep, texts):
            f.write(f"{n}\t{t}\n")
    print(f"wrote {args.out} ({len(keep)} lines)")


if __name__ == "__main__":
    main()
