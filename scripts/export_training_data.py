"""
export_training_data.py — build the W1 training npz (synthetic + real lines).

Attempt 1 was synthetic-only and the gate failed (the model memorized the
renderer). Attempt 2 mixes in real lines:
  * heiDATA dev books (human GT, real letterpress) - consumed as training data
    from here on, so the dev set stops being a tuning set
  * Gemini-labeled lines from v2 pages outside the frozen anchor
    (`scripts/label_real_lines.py`)
Real lines are repeated (`--real-repeat`) because they are the rare, high-value
part of the pool.

Usage:
    python scripts/export_training_data.py --out deva_crnn/data/train_v2.npz \
        --n 30000 --fonts all --aug-level heavy \
        --real-heidata data/doc_eval/heidata_dev \
        --real-dir data/doc_eval/deva_real_lines --real-repeat 3
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

import doc_data  # noqa: E402
from deva_crnn.augment import augment_line  # noqa: E402
from deva_crnn.data import export_npz  # noqa: E402
from deva_crnn.gate import heidata_line_crops  # noqa: E402
from doc_data import synthesize_deva_lines  # noqa: E402


def load_real_lines(real_dir: str):
    """(images, texts) from a label_real_lines.py output dir."""
    labels = os.path.join(real_dir, "labels.tsv")
    if not os.path.exists(labels):
        return [], []
    images, texts = [], []
    for line in open(labels, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        name, text = line.split("\t", 1)
        img = doc_data.imread_safe(os.path.join(real_dir, "lines", name))
        if img is None or not text:
            continue
        images.append(img)
        texts.append(text)
    return images, texts


def main():
    ap = argparse.ArgumentParser(description="Export W1 training data")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "deva_crnn", "data",
                                                  "train.npz"))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clean-frac", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fonts", choices=("default", "all"), default="default",
                    help="'all' = every installed Devanagari font + jitter")
    ap.add_argument("--aug-level", choices=("light", "heavy"), default="light")
    ap.add_argument("--real-dir", default=None,
                    help="dir produced by scripts/label_real_lines.py")
    ap.add_argument("--real-heidata", default=None,
                    help="heiDATA dev dataset dir (human-GT letterpress lines)")
    ap.add_argument("--real-repeat", type=int, default=1)
    ap.add_argument("--match-charset", default=None,
                    help="npz whose charset every label must fit (warm-start)")
    args = ap.parse_args()

    t0 = time.time()
    fonts = doc_data.available_deva_font_specs() if args.fonts == "all" else None
    images, texts = synthesize_deva_lines(
        args.n, seed=args.seed, fonts=fonts, jitter=args.fonts == "all")
    counts = {"synthetic": len(texts)}

    real_images, real_texts = [], []
    if args.real_heidata:
        hgts, hcrops = heidata_line_crops(args.real_heidata)
        real_images.extend(hcrops)
        real_texts.extend(hgts)
        counts["heidata_dev"] = len(hgts)
    if args.real_dir:
        r_imgs, r_txts = load_real_lines(args.real_dir)
        real_images.extend(r_imgs)
        real_texts.extend(r_txts)
        counts["real_gemini"] = len(r_txts)
    counts["real_repeat"] = args.real_repeat
    for _ in range(max(1, args.real_repeat)):
        images.extend(real_images)
        texts.extend(real_texts)
    counts["real_total"] = len(real_images) * max(1, args.real_repeat)
    counts["total"] = len(texts)

    rng = np.random.default_rng(args.seed + 1)
    mixed = [im if rng.random() < args.clean_frac
             else augment_line(im, rng, level=args.aug_level) for im in images]
    if args.match_charset:
        from deva_crnn.data import load_npz
        _imgs, base_texts = load_npz(args.match_charset)
        base = {c for t in base_texts for c in t}
        keep = [i for i, t in enumerate(texts) if not (set(t) - base)]
        counts["charset_filtered"] = len(texts) - len(keep)
        images = [mixed[i] for i in keep]
        texts = [texts[i] for i in keep]
    counts["total"] = len(texts)
    export_npz(images, texts, args.out, w=args.width)
    card = {
        "n": len(texts), "seed": args.seed, "clean_frac": args.clean_frac,
        "aug_level": args.aug_level, "fonts": args.fonts,
        "match_charset": args.match_charset,
        "font_files": sorted({s["path"] for s in (fonts or [])}),
        "counts": counts,
        "charset_size": len({c for t in texts for c in t}),
        "size_mb": round(os.path.getsize(args.out) / 1e6, 1),
        "source": ("doc_data.synthesize_deva_lines + deva_crnn.gate."
                   "heidata_line_crops + label_real_lines + augment_line"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.splitext(args.out)[0] + ".card.json", "w",
              encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"wrote {args.out}: {card} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
