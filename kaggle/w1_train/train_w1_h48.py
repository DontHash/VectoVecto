"""
train_w1_h48.py — Kaggle GPU kernel: W1 attempt 3b (input height 48).

Inputs (private datasets):
  * `bhishmbhandari/vectovecto-w1-training-data` — `train_v6_h48.npz` (+card)
  * `bhishmbhandari/vectovecto-w1-code` — flat `deva_crnn/*.py` sources

The Kaggle CLI's dataset uploader skips subdirectories by default, so the
package is assembled here from the flat sources before importing. Trains from
scratch, saves `/kaggle/working/w1_h48/{ckpt.pt,metrics.json}` and prints the
per-epoch history, so the log alone is enough to read the result.

No internet, no pip installs: the Kaggle GPU image ships torch/numpy/PIL.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from pathlib import Path

PKG = "deva_crnn"
INPUT = Path("/kaggle/input")


def tree(max_entries: int = 40) -> str:
    """Compact listing of the mounted inputs (for failure diagnosis)."""
    lines = []
    for p in sorted(INPUT.rglob("*"))[:max_entries]:
        kind = "d" if p.is_dir() else "f"
        size = "" if p.is_dir() else f" ({p.stat().st_size / 1e6:.1f} MB)"
        lines.append(f"  [{kind}] {p}{size}")
    return "\n".join(lines)


def find_input(*names: str) -> str:
    """Directory under /kaggle/input that contains any of `names`."""
    for root in sorted(INPUT.glob("*/")):
        for name in names:
            if glob.glob(os.path.join(root, name)):
                return str(root).rstrip("/")
    raise SystemExit(f"no dataset with {names} mounted under /kaggle/input\n"
                     f"tree:\n{tree()}")


def find_npz() -> str:
    """Recursive search: the mount layout is not worth guessing twice."""
    hits = sorted(INPUT.rglob("*.npz"))
    if not hits:
        raise SystemExit(f"no npz under /kaggle/input\ntree:\n{tree()}")
    print(f"[kaggle] npz candidates: {[str(h) for h in hits[:3]]}", flush=True)
    return str(max(hits, key=lambda p: p.stat().st_size))


def assemble_package(code_dir: str) -> str:
    """Flat .py sources -> importable package (or reuse an existing dir)."""
    existing = os.path.join(code_dir, PKG)
    if os.path.isdir(existing):
        return code_dir
    work = "/kaggle/working/pkg"
    pkg_dir = os.path.join(work, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    srcs = [p for p in glob.glob(os.path.join(code_dir, "*.py"))]
    if not srcs:
        raise SystemExit(f"no .py sources in {code_dir}")
    for src in srcs:
        shutil.copy(src, pkg_dir)
    init = os.path.join(pkg_dir, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    print(f"[kaggle] package assembled at {pkg_dir} ({len(srcs)} modules)",
          flush=True)
    return work


def main() -> None:
    import torch

    code_dir = find_input(f"{PKG}/data.py", "data.py", "train.py")
    data = find_npz()
    sys.path.insert(0, assemble_package(code_dir))

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
