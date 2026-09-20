"""
kaggle_train.py — Kaggle T4 notebook script for Tier B training.

HOW TO USE (Kaggle):
1. Create a new Kaggle Notebook.
2. Add the dataset "DIV2K Dataset - Train Data (1GB)" via Add Input -> Datasets
   -> search "div2k". This mounts /kaggle/input/div2k-dataset/ in the kernel.
3. Add this repo's files (drunet.py, deep_unfolding.py, degradation.py,
   perceptual_loss.py, discriminator.py, train_deep_sr.py) via Add Input ->
   New Dataset -> upload them, OR paste their contents into cells.
4. Set the accelerator to GPU (T4 x2). Set Internet ON for the VGG download
   (~548MB, one-time first epoch) and tqdm install.
5. Run all cells. Checkpoints land in /kaggle/working/deep_sr/.
6. After training, in the right panel -> Output -> download
   best_checkpoint.pth. Drop it into your repo at
   artifacts/deep_sr/best_checkpoint.pth and inference.py auto-uses it.

This script is self-contained — it installs deps, points train_deep_sr at
the Kaggle paths, and launches training with T4-tuned defaults.
"""

# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import subprocess
import sys


def _cell(msg):
    print(f"\n[KAGGLE] === {msg} ===", flush=True)


def main():
    # 1. Install deps (internet must be ON)
    _cell("Installing dependencies")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "tqdm", "opencv-python", "scikit-image"], check=False)

    _cell("Verifying environment")
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"VRAM: free {free/1e9:.2f} GB / total {total/1e9:.2f} GB")
    if not torch.cuda.is_available():
        print("FATAL: GPU not enabled. Settings -> Accelerator -> GPU T4 x2.",
              file=sys.stderr)
        sys.exit(1)

    # 2. Confirm DIV2K is attached
    _cell("Locating DIV2K")
    div2k_root = "/kaggle/input"
    found_dir = None
    if os.path.isdir(div2k_root):
        for entry in os.listdir(div2k_root):
            base = os.path.join(div2k_root, entry)
            if os.path.isdir(base):
                for sub in ("DIV2K_train_HR", "DIV2K_train_HR/DIV2K_train_HR", "train_HR"):
                    p = os.path.join(base, sub)
                    if os.path.isdir(p):
                        imgs = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if imgs:
                            found_dir = p
                            break
            if found_dir: break

    if found_dir is None:
        print("FATAL: DIV2K HR not found. Add the 'div2k-dataset' input to the notebook.",
              file=sys.stderr)
        print("Search 'div2k' in Add Input -> Datasets.", file=sys.stderr)
        sys.exit(1)
    print(f"DIV2K HR: {found_dir} ({len(os.listdir(found_dir))} files)")
    os.environ["DIV2K_HR_DIR"] = found_dir

    # 3. Output dir for checkpoints (Kaggle saves /kaggle/working/* on exit)
    out_dir = "/kaggle/working/deep_sr"
    os.makedirs(out_dir, exist_ok=True)
    os.environ["ARTIFACTS_DIR"] = out_dir
    print(f"Checkpoints -> {out_dir}")

    # 4. Run training.
    _cell("Smoke test (1 iter) before launching real training")
    subprocess.run([sys.executable, "train_deep_sr.py", "--smoke", "--device", "cuda"],
                   check=False)

    _cell("LAUNCHING T4 TRAINING")
    # Patch 32, batch 8, 100 epochs of 250 iters = 25k iters total.
    # T4 has 16GB so this fits comfortably. Adjust --epochs up for more quality.
    cmd = [
        sys.executable, "train_deep_sr.py",
        "--device", "cuda",
        "--epochs", "100",
        "--batch", "8",
        "--patch", "32",
        "--unfolding-iters", "5",
        "--iters-per-epoch", "250",
        "--save-every", "2000",
        "--log-every", "50",
        "--workers", "2",      # Kaggle kernels allow 2-4 workers
    ]
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=False)

    _cell("Training done. Outputs:")
    for f in os.listdir(out_dir):
        p = os.path.join(out_dir, f)
        size_mb = os.path.getsize(p) / 1e6
        print(f"  {f}: {size_mb:.1f} MB")
    print("\nDownload these from the notebook 'Output' panel on the right.")


if __name__ == "__main__":
    main()