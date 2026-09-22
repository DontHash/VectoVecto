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


def beam_search_decode(log_probs: np.ndarray, charset: List[str],
                       beam_width: int = 8) -> str:
    """CTC prefix beam search over (T, C) log-probabilities.

    Greedy argmax commits to one alignment per frame; on noisy letterpress
    digits the second-best prefix is often the right one, and beam search
    keeps those alternatives. `charset[0]` is the CTC blank.
    """
    beams = {(): (0.0, -np.inf)}  # prefix -> (p_blank, p_nonblank) log-probs
    for t in range(log_probs.shape[0]):
        lp = log_probs[t]
        nxt: dict = {}
        for prefix, (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)
            for c in range(lp.shape[0]):
                p = lp[c]
                if c == 0:  # blank: prefix stays
                    cur = nxt.get(prefix, (-np.inf, -np.inf))
                    nxt[prefix] = (np.logaddexp(cur[0], p_total + p), cur[1])
                    continue
                last = prefix[-1] if prefix else None
                if c == last:  # repeat: only extend via a blank in between
                    cur = nxt.get(prefix, (-np.inf, -np.inf))
                    nxt[prefix] = (cur[0], np.logaddexp(cur[1], pnb + p))
                    new_prefix = prefix + (c,)
                    cur = nxt.get(new_prefix, (-np.inf, -np.inf))
                    nxt[new_prefix] = (cur[0],
                                       np.logaddexp(cur[1], pb + p))
                else:
                    new_prefix = prefix + (c,)
                    cur = nxt.get(new_prefix, (-np.inf, -np.inf))
                    nxt[new_prefix] = (cur[0],
                                       np.logaddexp(cur[1], p_total + p))
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1])
                            )[:beam_width])
    best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))
    return "".join(charset[c] for c in best[0])


def recognize_lines(model, charset: List[str], images_bgr: List[np.ndarray],
                    device: str = "cpu", decode_mode: str = "greedy",
                    beam_width: int = 8) -> List[str]:
    if not images_bgr:
        return []
    batch = np.stack([normalize_line(im) for im in images_bgr]).astype(np.float32)
    x = torch.from_numpy(batch / 255.0).unsqueeze(1)
    x = ((x - 0.5) / 0.5).to(device)
    with torch.no_grad():
        logits = model(x)
        if decode_mode == "beam":
            log_probs = logits.log_softmax(-1).cpu().numpy()  # (T, B, C)
            return [beam_search_decode(log_probs[:, b, :], charset,
                                       beam_width=beam_width)
                    for b in range(log_probs.shape[1])]
        preds = logits.argmax(-1).permute(1, 0)
    return [decode(p.tolist(), charset) for p in preds]


def main():
    ap = argparse.ArgumentParser(description="deva_crnn line inference")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode", choices=("greedy", "beam"), default="greedy")
    ap.add_argument("--beam-width", type=int, default=8)
    args = ap.parse_args()

    model, charset = load_model(args.ckpt, args.device)
    names = sorted(f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    imgs = [cv2.imread(os.path.join(args.images_dir, n)) for n in names]
    keep = [(n, im) for n, im in zip(names, imgs) if im is not None]
    texts = recognize_lines(model, charset, [im for _, im in keep], args.device,
                            decode_mode=args.decode, beam_width=args.beam_width)
    with open(args.out, "w", encoding="utf-8") as f:
        for (n, _), t in zip(keep, texts):
            f.write(f"{n}\t{t}\n")
    print(f"wrote {args.out} ({len(keep)} lines)")


if __name__ == "__main__":
    main()
