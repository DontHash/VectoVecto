"""
vlm_eval.py — Kaggle T4 kernel: Qwen3-VL-8B NF4 bake-off (Phase 1).

Input: private dataset `bhishmbhandari/vectovecto-vlm-eval-data` holding
code.tar.gz (flat repo subset) + data.tar.gz (frozen doc_eval dirs). Kaggle
auto-extracts the archives, so the mounted dataset presents `code/` and
`data/` trees; both are copied into a writable repo root.

Runtime 4-bit NF4 of the Apache-2.0 checkpoint — the pre-quantized unsloth
repos trip bitsandbytes' state loader under transformers 5.x. Pages first
(the decisive number), then lines. Outputs land in /kaggle/working.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = Path("/kaggle/working/repo")
WORK = Path("/kaggle/working")
DATASET_SLUG = "vectovecto-vlm-eval-data"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def dataset_root() -> Path:
    hits = [p for p in Path("/kaggle/input").rglob(DATASET_SLUG) if p.is_dir()]
    if hits:
        return hits[0]
    listing = "\n".join(str(p) for p in
                        sorted(Path("/kaggle/input").rglob("*"))[:60])
    raise SystemExit(f"dataset {DATASET_SLUG} not mounted:\n{listing}")


def stage_repo() -> None:
    root = dataset_root()
    print("[kaggle] dataset root:", root, flush=True)
    REPO.mkdir(parents=True, exist_ok=True)
    for sub in ("code", "data"):
        src = root / sub
        if src.is_dir():  # Kaggle auto-extracted the archive
            shutil.copytree(src, REPO, dirs_exist_ok=True)
            continue
        tar = root / f"{sub}.tar.gz"
        if not tar.is_file():
            raise SystemExit(f"neither {src} nor {tar} exists")
        with tarfile.open(tar) as tf:
            tf.extractall(REPO)
    print("[kaggle] repo tree:", sorted(p.name for p in REPO.iterdir()),
          flush=True)


def main() -> None:
    sh("nvidia-smi")
    sh('pip install -q --upgrade transformers accelerate bitsandbytes '
       'opencv-python-headless jiwer Pillow "jinja2>=3.1" rapidocr '
       'onnxruntime')
    stage_repo()

    py = sys.executable
    runs = (
        ("pages", f"{py} evals/harness/eval_models.py "
                  f"--model qwen3vl-8b-4bit-rt --mode pages "
                  f"--data-dir data/doc_eval/nepali_pdf_v2 --limit 10 "
                  f"--json {WORK}/qwen8b4bit_pages.json"),
        ("lines", f"{py} evals/harness/eval_models.py "
                  f"--model qwen3vl-8b-4bit-rt --mode lines "
                  f"--lines-source deva_real_lines --limit 150 "
                  f"--json {WORK}/qwen8b4bit_lines.json"),
    )
    for name, cmd in runs:
        r = subprocess.run(cmd, cwd=REPO, shell=True)
        print(f"[kaggle] {name} exit={r.returncode}", flush=True)


if __name__ == "__main__":
    main()
