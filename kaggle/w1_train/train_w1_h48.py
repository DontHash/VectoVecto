"""
train_w1_h48.py — Kaggle GPU kernel: W1 attempt 3b (input height 48).

Runs from the private dataset `bhishmbhandari/vectovecto-w1-training-data`
(deva_crnn package + train_v6_h48.npz). Trains from scratch, saves
`/kaggle/working/w1_h48/{ckpt.pt,metrics.json}` for download, and prints the
per-epoch history so the log alone is enough to read the result.

No internet, no pip installs: the Kaggle GPU image ships torch/numpy/PIL.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def find_dataset_dir() -> str:
    """The dataset mounts at /kaggle/input/<slug>; find the one with our pkg."""
    for cand in sorted(glob.glob("/kaggle/input/*/")):
        if os.path.isdir(os.path.join(cand, "deva_crnn")):
            return cand.rstrip("/")
    raise SystemExit("deva_crnn dataset not mounted under /kaggle/input")


def main() -> None:
    import torch

    data_dir = find_dataset_dir()
    sys.path.insert(0, data_dir)
    npz = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz:
        raise SystemExit(f"no npz found in {data_dir}")
    data = npz[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[kaggle] data={data} ({os.path.getsize(data) / 1e6:.0f} MB) "
          f"device={device} torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"[kaggle] gpu={torch.cuda.get_device_name(0)}", flush=True)

    from deva_crnn.train import train

    out_dir = "/kaggle/working/w1_h48"
    result = train(data, out_dir, epochs=26, batch=64, lr=1e-3, val_split=0.05,
                   seed=1, max_hours=3.0, workers=2, in_h=48)

    history = result["history"]
    best = max(history, key=lambda r: r.get("exact_match") or 0.0)
    summary = {"epochs_done": len(history), "final": history[-1],
               "best_val": best, "out_dir": out_dir,
               "artifacts": sorted(os.listdir(out_dir))}
    print("[kaggle] SUMMARY " + json.dumps(summary, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
