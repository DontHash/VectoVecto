"""
train_w1.py — Kaggle GPU kernel: config-driven Devanagari line training.

Inputs (private datasets):
  * `bhishmbhandari/vectovecto-w1-code` — flat `deva_crnn/*.py` sources,
    `w1_config.json`, optional checkpoints
  * one or more `*-training-data` / `*-data-*` datasets holding the npz

The config selects everything (npz filename, height/width, epochs, lr, init),
so a new experiment is a config change, not a code edit:

    {"npz": "train_v8_h48.npz", "in_h": 48, "in_w": 256, "epochs": 26,
     "batch": 64, "lr": 0.001, "init": null, "out_dir": "w1_s1",
     "lr_schedule": "none", "save_best": true, "seed": 1}

The Kaggle CLI's dataset uploader skips subdirectories, so the package is
assembled here from the flat sources. No internet, no pip installs.
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
CONFIG_NAME = "w1_config.json"
DEFAULTS = {"npz": None, "in_h": 48, "in_w": 256, "epochs": 26, "batch": 64,
            "lr": 1e-3, "init": None, "out_dir": "w1_run",
            "lr_schedule": "none", "save_best": True, "seed": 1,
            "val_split": 0.05, "max_hours": 3.0, "workers": 2}


def tree(max_entries: int = 40) -> str:
    """Compact listing of the mounted inputs (for failure diagnosis)."""
    lines = []
    for p in sorted(INPUT.rglob("*"))[:max_entries]:
        kind = "d" if p.is_dir() else "f"
        size = "" if p.is_dir() else f" ({p.stat().st_size / 1e6:.1f} MB)"
        lines.append(f"  [{kind}] {p}{size}")
    return "\n".join(lines)


def load_config() -> dict:
    hits = sorted(INPUT.rglob(CONFIG_NAME))
    if not hits:
        raise SystemExit(f"no {CONFIG_NAME} under /kaggle/input\ntree:\n{tree()}")
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(hits[0].read_text(encoding="utf-8")))
    print(f"[kaggle] config from {hits[0]}: {json.dumps(cfg)}", flush=True)
    return cfg


def find_file(name: str, pattern: str) -> str:
    hits = sorted(INPUT.rglob(pattern))
    if not hits:
        raise SystemExit(f"no {pattern} under /kaggle/input\ntree:\n{tree()}")
    if name:
        exact = [h for h in hits if h.name == name]
        if not exact:
            raise SystemExit(f"{name} not found; candidates: "
                             f"{[h.name for h in hits]}")
        return str(exact[0])
    return str(max(hits, key=lambda p: p.stat().st_size))


def assemble_package(code_dir: str) -> str:
    """Flat .py sources -> importable package (or reuse an existing dir)."""
    existing = os.path.join(code_dir, PKG)
    if os.path.isdir(existing):
        return code_dir
    work = "/kaggle/working/pkg"
    pkg_dir = os.path.join(work, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    srcs = glob.glob(os.path.join(code_dir, "*.py"))
    if not srcs:
        raise SystemExit(f"no .py sources in {code_dir}")
    for src in srcs:
        shutil.copy(src, pkg_dir)
    open(os.path.join(pkg_dir, "__init__.py"), "a").close()
    print(f"[kaggle] package assembled at {pkg_dir} ({len(srcs)} modules)",
          flush=True)
    return work


def main() -> None:
    import torch

    cfg = load_config()
    code_dir = None
    for root in sorted(INPUT.glob("*/")):
        if glob.glob(os.path.join(root, "*.py")):
            code_dir = str(root).rstrip("/")
            break
    if code_dir is None:
        raise SystemExit(f"no code dataset under /kaggle/input\ntree:\n{tree()}")
    sys.path.insert(0, assemble_package(code_dir))

    data = find_file(cfg["npz"], "*.npz")
    init = find_file(cfg["init"], "*.pt") if cfg["init"] else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[kaggle] data={data} ({os.path.getsize(data) / 1e6:.0f} MB) "
          f"init={init} device={device} torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"[kaggle] gpu={torch.cuda.get_device_name(0)}", flush=True)

    from deva_crnn.train import train

    out_dir = f"/kaggle/working/{cfg['out_dir']}"
    result = train(data, out_dir, epochs=cfg["epochs"], batch=cfg["batch"],
                   lr=cfg["lr"], val_split=cfg["val_split"], seed=cfg["seed"],
                   max_hours=cfg["max_hours"], workers=cfg["workers"],
                   in_h=cfg["in_h"], in_w=cfg["in_w"], init=init,
                   lr_schedule=cfg["lr_schedule"], save_best=cfg["save_best"])

    history = result["history"]
    best = max(history, key=lambda r: r.get("exact_match") or 0.0)
    summary = {"config": cfg, "mode": "finetune" if init else "scratch",
               "epochs_done": len(history), "final": history[-1],
               "best_val": best, "out_dir": out_dir,
               "artifacts": sorted(os.listdir(out_dir))}
    print("[kaggle] SUMMARY " + json.dumps(summary, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
