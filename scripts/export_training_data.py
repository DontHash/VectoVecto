"""
export_training_data.py — build the W1 training npz (synthetic, augmented).

Usage:
    python scripts/export_training_data.py --out deva_crnn/data/train.npz \
        --n 30000 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from deva_crnn.augment import augment_line
from deva_crnn.data import export_npz  # noqa: E402
from doc_data import synthesize_deva_lines  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Export W1 synthetic training data")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "deva_crnn", "data",
                                                  "train.npz"))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clean-frac", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    images, texts = synthesize_deva_lines(args.n, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    mixed = [im if rng.random() < args.clean_frac else augment_line(im, rng)
             for im in images]
    export_npz(mixed, texts, args.out, w=args.width)
    card = {
        "n": len(texts), "seed": args.seed, "clean_frac": args.clean_frac,
        "charset_size": len({c for t in texts for c in t}),
        "size_mb": round(os.path.getsize(args.out) / 1e6, 1),
        "source": "doc_data.synthesize_deva_lines + deva_crnn.data.augment_line",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.splitext(args.out)[0] + ".card.json", "w",
              encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"wrote {args.out}: {card}", flush=True)


if __name__ == "__main__":
    main()
