"""
train_deep_sr.py — Tier B training: Deep Unfolding SR with DRUNet, VGG
perceptual loss, and PatchGAN adversarial loss, on Real-ESRGAN second-order
degradation.

This replaces train.py's smoke-test (50 synthetic circles, 2 epochs, L1) with
a competitive setup:

  Generator  : DeepUnfoldingSR(DRUNet) — unrolled proximal gradient descent
               with a 20-layer ResNet learned prior.
  Losses     : lambda_1 * L1  +  lambda_perc * VGG19  +  lambda_adv * PatchGAN
               +  lambda_tv * Total Variation smoothing.
  Data       : DIV2K (800 HR images) → degraded via Real-ESRGAN 2nd-order.
  Optimizer  : Adam (G_lr=1e-4, D_lr=1e-4), warmup + decay 0.5 every 50k iters.
  Scale      : 4× (matches the existing esrgan_inference paths).

Target hardware: Kaggle T4 GPU (16GB). Patches = 128, batch = 8 fits.
On CPU there is no point — the script detects and warns + reduces iters, but
running on CPU is a proof-of-concept only.

Run:

    python prepare_div2k.py
    python train_deep_sr.py --epochs 100 --batch 8 --patch 128 --device cuda

Checkpoints land in ./artifacts/deep_sr/ (best + latest).

Smoke run (no DIV2K, synthetic data, 1 step) to verify all components compose:

    python train_deep_sr.py --smoke
"""

# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import os
import sys
import time
import random
import signal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

from drunet import DRUNet
from deep_unfolding import DeepUnfoldingSR
from degradation import degrade_known_kernel
from perceptual_loss import VGGPerceptualLoss
from discriminator import PatchDiscriminator, gan_loss_generator, gan_loss_discriminator

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "deep_sr")
DIV2K_HR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "DIV2K", "DIV2K_train_HR")


def _auto_detect_div2k():
    """
    Locate DIV2K HR training images across environments:
      1. Kaggle dataset mount: /kaggle/input/div2k-dataset/...
      2. Local prepare_div2k.py output: ./data/DIV2K/DIV2K_train_HR
      3. Env override: DIV2K_HR_DIR env var
    Returns the first HR directory that contains image files, or None.
    """
    candidates = []

    # 3. Explicit env var
    env_dir = os.environ.get("DIV2K_HR_DIR")
    if env_dir:
        candidates.append(env_dir)

    # 1. Kaggle: the dataset shows up under /kaggle/input/. The DIV2K Kaggle
    #    dataset names its subdirs DIV2K_train_HR / DIV2K_valid_HR (matches local).
    if os.path.isdir("/kaggle/input"):
        for entry in os.listdir("/kaggle/input"):
            base = os.path.join("/kaggle/input", entry)
            if os.path.isdir(base):
                for sub in ("DIV2K_train_HR", "DIV2K_train_HR/DIV2K_train_HR", "train_HR"):
                    p = os.path.join(base, sub)
                    if os.path.isdir(p):
                        candidates.append(p)

    # 2. Local prepare_div2k.py
    candidates.append(DIV2K_HR_DIR)

    for c in candidates:
        if os.path.isdir(c):
            imgs = [f for f in os.listdir(c) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                return c
    return None


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class DIV2KSRDataset(Dataset):
    """
    Reads HR patches from DIV2K at random crops of size `patch*scale`, then
    degrades them online via degrade_real_esrgan to produce matched LR-HR pairs.
    """

    def __init__(self, hr_dir, patch=128, scale=4, length_per_epoch=4000):
        self.hr_dir = hr_dir
        self.patch = patch  # patch in the LR space (output HR will be patch*scale)
        self.scale = scale
        self.length = length_per_epoch
        self.files = []
        if os.path.isdir(hr_dir):
            for f in sorted(os.listdir(hr_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    self.files.append(os.path.join(hr_dir, f))
        if not self.files:
            self.files = None  # signal: synthetic fallback in __getitem__

    def __len__(self):
        return self.length

    def _random_hr_crop(self, img, target_h, target_w):
        """Random crop a H,W crop from a uint8 H,W,3 image, padded if too small."""
        h, w = img.shape[:2]
        if h < target_h or w < target_w:
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            h, w = img.shape[:2]
        y = random.randint(0, h - target_h)
        x = random.randint(0, w - target_w)
        return img[y:y + target_h, x:x + target_w]

    def __getitem__(self, idx):
        hr_size = self.patch * self.scale
        if self.files is None:
            # Synthetic fallback: random colors/gradients + shapes — for smoke test.
            img = np.zeros((hr_size, hr_size, 3), dtype=np.uint8)
            for c in range(3):
                v = random.randint(0, 255)
                img[:, :, c] = v
            for _ in range(random.randint(2, 6)):
                cv2.circle(img, (random.randint(0, hr_size - 1), random.randint(0, hr_size - 1)),
                           random.randint(5, 50),
                           (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
        else:
            path = random.choice(self.files)
            img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
            if img is None:
                img = np.zeros((hr_size, hr_size, 3), dtype=np.uint8)
            img = self._random_hr_crop(img, hr_size, hr_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hr = img.astype(np.float32) / 255.0
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        # Self-consistent degradation: returns the ACTUAL blur kernel used so
        # the unfolding's data-fidelity term matches how y was made.
        lr_t, sigma_noise, kernel = degrade_known_kernel(hr_t, scale=self.scale)
        return {
            "hr": hr_t.squeeze(0),
            "lr": lr_t.squeeze(0),
            "sigma": torch.tensor(float(sigma_noise)),  # scalar
            "kernel": kernel,                            # (K,K) tensor
        }


# ----------------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------------
def _total_variation(img):
    """Anisotropic TV. img: (B,C,H,W)."""
    diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


def _checkpoint(state, path):
    # Atomic write: a Kaggle session can be killed at any moment, and a
    # half-written .pth would be unusable AND would poison --resume.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Smoke test (one forward/backward step on synthetic data; no DIV2K needed)
# ----------------------------------------------------------------------------
def smoke_test(device):
    print("=== SMOKE TEST (1 iter, synthetic, no DIV2K) ===")
    generator = DeepUnfoldingSR(DRUNet(in_channels=3), iterations=2, scale=4).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)
    vgg = VGGPerceptualLoss().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    hr = torch.rand(1, 3, 256, 256, device=device)
    lr, sigma, kernel = degrade_known_kernel(hr.cpu(), scale=4)
    lr = lr.to(device); sigma = torch.tensor(float(sigma), device=device)
    kernel = kernel.to(device)

    # Generator step
    generator.train()
    sr = generator(lr, kernel, sigma_noise=sigma)
    # Align SR to HR size (degradation may not produce an exact HR/4 -> HR crop)
    sr = sr[..., :hr.shape[-2], :hr.shape[-1]]
    hr_a = hr[..., :sr.shape[-2], :sr.shape[-1]]
    l1 = F.l1_loss(sr, hr_a)
    perc = vgg(sr.clamp(0, 1), hr_a)
    fake_logits = discriminator(sr)
    adv = gan_loss_generator(fake_logits)
    tv = _total_variation(sr)
    g_loss = l1 + 0.1 * perc + 0.005 * adv + 0.001 * tv
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()

    # Discriminator step
    with torch.no_grad():
        sr_fake = generator(lr, kernel, sigma_noise=sigma).detach()[..., :hr.shape[-2], :hr.shape[-1]]
    real_logits = discriminator(hr_a)
    fake_logits = discriminator(sr_fake)
    d_loss = gan_loss_discriminator(real_logits, fake_logits)
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()

    print(f"G loss: l1={l1.item():.4f} perc={perc.item():.4f} adv={adv.item():.4f} tv={tv.item():.4f}")
    print(f"D loss: {d_loss.item():.4f}")
    print(f"SR shape: {sr.shape} | HR shape: {hr_a.shape}")
    print(" === SMOKE OK === ")


# ----------------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------------
class _TimeBudgetReached(Exception):
    """Raised internally to unwind the training loop cleanly on --time-budget-min."""


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def train(args):
    device = torch.device(args.device)
    if device.type == "cpu":
        print("WARNING: training on CPU is very slow. Tier B expects a GPU. "
              "Use --smoke on CPU; for real training, use --device cuda.", file=sys.stderr)
        if not args.smoke and not args.allow_cpu:
            print("Pass --allow-cpu to override. Aborting.", file=sys.stderr)
            sys.exit(1)

    if args.smoke:
        return smoke_test(device)

    # Artifact output dir: default local, override for Kaggle (/kaggle/working/).
    # The Kaggle notebook sets ARTIFACTS_DIR env so checkpoints land in Kaggle's
    # downloadable output path.
    artifacts_dir = os.environ.get("ARTIFACTS_DIR") or args.artifacts_dir or ARTIFACTS
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {artifacts_dir}")

    _set_seed(args.seed)

    # Auto-detect DIV2K HR directory across environments (Kaggle / local / env).
    hr_dir = _auto_detect_div2k()
    if hr_dir is None:
        print("ERROR: DIV2K HR images not found in any known location.", file=sys.stderr)
        print("On Kaggle: attach the 'div2k-dataset' to the notebook.", file=sys.stderr)
        print("Locally:   run `python prepare_div2k.py` first.", file=sys.stderr)
        print("Or:        set DIV2K_HR_DIR env var to the HR folder.", file=sys.stderr)
        sys.exit(1)
    print(f"Using DIV2K HR from: {hr_dir}")

    dataset = DIV2KSRDataset(hr_dir, patch=args.patch, scale=args.scale,
                             length_per_epoch=args.iters_per_epoch * args.batch)

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # Build networks
    generator = DeepUnfoldingSR(DRUNet(in_channels=3, num_feat=64, num_blocks=20),
                                iterations=args.unfolding_iters, scale=args.scale,
                                step_size=0.2).to(device)
    discriminator = PatchDiscriminator(in_channels=3, num_feat=64).to(device)
    vgg = VGGPerceptualLoss().to(device)
    vgg.eval()

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.99))

    # Loss weights (Real-ESRGAN convention)
    w_l1, w_perc, w_adv, w_tv = 1.0, args.w_perceptual, args.w_adv, args.w_tv

    scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=args.decay_step, gamma=0.5)

    # --- Resume from checkpoint if requested ---
    # Without this, restarting a Kaggle kernel starts from scratch and
    # WASTES all previous training (observed: 6000 iters lost on restart).
    start_iter = 0
    start_epoch = 0
    if args.resume and os.path.exists(os.path.join(artifacts_dir, "latest_checkpoint.pth")):
        ckpt_path = os.path.join(artifacts_dir, "latest_checkpoint.pth")
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt["model_state_dict"])
        discriminator.load_state_dict(ckpt["d_state_dict"])
        start_iter = ckpt.get("iter", 0)
        start_epoch = start_iter // args.iters_per_epoch
        best_loss = ckpt.get("g_loss", float("inf"))
        print(f"  Resumed at iter {start_iter}, epoch {start_epoch}, best_loss {best_loss:.4f}")
        # Advance scheduler to the right step
        for _ in range(start_epoch):
            scheduler_g.step()
    else:
        best_loss = float("inf")

    total_iters = args.iters_per_epoch * args.epochs
    print(f"\nTraining: scale={args.scale} unfolding_iters={args.unfolding_iters} "
          f"patch={args.patch} batch={args.batch} iters={total_iters} "
          f"(resuming from iter {start_iter})")
    iter_count = start_iter
    t_lo = time.time()
    train_t0 = time.time()

    def _save_now(iter_n, loss_val, tag=""):
        """Save latest; promote to best only if improved. Used by periodic saves,
        time-budget exits, and interrupt handlers."""
        nonlocal best_loss
        state = {
            "iter": iter_n,
            "model_state_dict": generator.state_dict(),
            "d_state_dict": discriminator.state_dict(),
            "g_loss": loss_val,
        }
        _checkpoint(state, os.path.join(artifacts_dir, "latest_checkpoint.pth"))
        if loss_val < best_loss:
            best_loss = loss_val
            _checkpoint(state, os.path.join(artifacts_dir, "best_checkpoint.pth"))
            print(f"   saved best (G={best_loss:.4f}){tag}")

    last_g_loss = best_loss if best_loss != float("inf") else 0.0
    try:
        for epoch in range(start_epoch, args.epochs):
            generator.train()
            for batch_idx, batch in enumerate(loader):
                hr = batch["hr"].to(device, non_blocking=True)
                lr = batch["lr"].to(device, non_blocking=True)
                sigma = batch["sigma"].to(device, non_blocking=True)
                # Per-sample degradation kernels: (B,K,K) -> (B,1,1,K,K)
                kernel_b = batch["kernel"].to(device).unsqueeze(1).unsqueeze(1)

                # ---- Generator step ----
                sr = generator(lr, kernel_b, sigma_noise=sigma)
                # Align SR to HR (degradation may not yield exactly HR/scale);
                # we crop the larger side down to the smaller, on all spatial dims.
                min_h = min(sr.shape[-2], hr.shape[-2])
                min_w = min(sr.shape[-1], hr.shape[-1])
                sr = sr[..., :min_h, :min_w]
                hr_a = hr[..., :min_h, :min_w]
                sr_clamped = sr.clamp(0, 1)
                l1_loss = F.l1_loss(sr, hr_a)
                perc_loss = vgg(sr_clamped, hr_a)
                fake_logits = discriminator(sr)
                adv_loss = gan_loss_generator(fake_logits)
                tv_loss = _total_variation(sr)
                g_loss = w_l1 * l1_loss + w_perc * perc_loss + w_adv * adv_loss + w_tv * tv_loss
                opt_g.zero_grad(); g_loss.backward()
                # CRITICAL: unrolled optimization (5 data steps x 5 denoise steps)
                # compounds gradients -> norms explode 10-100x per iteration without
                # clipping (observed tail_grad_norm 2.8k -> 246k over 5 iters).
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                opt_g.step()

                # ---- Discriminator step (every K G steps for stability) ----
                if iter_count % args.d_every == 0:
                    with torch.no_grad():
                        sr_fake = generator(lr, kernel_b, sigma_noise=sigma).detach()[..., :min_h, :min_w]
                    real_logits = discriminator(hr_a)
                    fake_logits = discriminator(sr_fake)
                    d_loss = gan_loss_discriminator(real_logits, fake_logits)
                    opt_d.zero_grad(); d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    opt_d.step()

                iter_count += 1
                last_g_loss = g_loss.item()
                if iter_count % args.log_every == 0:
                    dt = time.time() - t_lo
                    t_lo = time.time()
                    print(f"[e{epoch:03d} i{iter_count:06d}] "
                          f"G={g_loss.item():.4f} (L1={l1_loss.item():.4f} "
                          f"perc={perc_loss.item():.4f} adv={adv_loss.item():.4f} "
                          f"tv={tv_loss.item():.4f}) D={d_loss.item():.4f} "
                          f"[{dt/args.log_every:.2f}s/it]")

                # ---- Checkpoint ----
                if iter_count % args.save_every == 0:
                    _save_now(iter_count, g_loss.item())

                # ---- Time budget (Kaggle session limit) ----
                # Stop cleanly BEFORE Kaggle kills the session, so progress is
                # always persisted instead of dying mid-epoch.
                if args.time_budget_min > 0 and (time.time() - train_t0) >= args.time_budget_min * 60:
                    print(f"\n[budget] {args.time_budget_min:g} min reached at iter {iter_count}; "
                          f"saving and exiting cleanly.")
                    _save_now(iter_count, g_loss.item(), tag=" (budget exit)")
                    raise _TimeBudgetReached

            scheduler_g.step()
    except _TimeBudgetReached:
        print("Stopped by time budget. Resume with --resume to continue.")
        return
    except KeyboardInterrupt:
        # Ctrl-C or SIGTERM (Kaggle "stop session") -> persist before dying.
        print("\nInterrupted -> saving latest checkpoint before exit...")
        _save_now(iter_count, last_g_loss, tag=" (interrupt)")
        print(f"Checkpoint saved at iter {iter_count}. Resume with --resume.")
        return

    print("Training complete. Final checkpoint saved.")
    _save_now(iter_count, last_g_loss, tag=" (final)")
    _checkpoint({
        "iter": iter_count,
        "model_state_dict": generator.state_dict(),
        "d_state_dict": discriminator.state_dict(),
    }, os.path.join(artifacts_dir, "final_checkpoint.pth"))


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Tier B training: Deep Unfolding SR")
    p.add_argument("--smoke", action="store_true", help="Run 1-iter synthetic smoke test and exit")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest_checkpoint.pth in artifacts dir. "
                        "CRITICAL for Kaggle: without this, restarting the kernel "
                        "starts from scratch and wastes all previous training.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--allow-cpu", action="store_true", help="Run full training on CPU (very slow)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--unfolding-iters", type=int, default=5, help="Deep unfolding K")
    p.add_argument("--patch", type=int, default=32, help="LR patch size (HR will be patch*scale)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--iters-per-epoch", type=int, default=250)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--decay-step", type=int, default=50000)
    p.add_argument("--w-perceptual", type=float, default=0.5,
                   help="VGG perceptual weight. ESRGAN-family uses ~1.0; 0.1 was too weak "
                        "-> output stayed soft. Sharpness comes mostly from this + GAN.")
    p.add_argument("--w-adv", type=float, default=0.02,
                   help="Adversarial weight. 0.005 was negligible -> GAN contributed ~nothing. "
                        "0.02-0.05 gives the generator real texture pressure. "
                        "Bump to 0.05 for a GAN finetune once structure converges.")
    p.add_argument("--w-tv", type=float, default=0.001,
                   help="Total-variation smoothness (anti-sharpening, keep small).")
    p.add_argument("--d-every", type=int, default=2, help="Run D step every K G steps (GAN stability)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--time-budget-min", type=float, default=0.0,
                   help="Stop cleanly after N minutes and save a checkpoint (0 = no limit). "
                        "Kaggle kills sessions at ~12h: set ~600 so every run persists its "
                        "progress and can --resume next session.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--artifacts-dir", default=None,
                   help="Where to save checkpoints. Default: ./artifacts/deep_sr. "
                        "On Kaggle, set to /kaggle/working/deep_sr or use ARTIFACTS_DIR env.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Kaggle/supervisors stop sessions with SIGTERM; route it through the same
    # save-then-exit path as Ctrl-C so no training progress is lost.
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    except (ValueError, OSError, AttributeError):
        pass  # not supported on this platform/thread
    train(args)


if __name__ == "__main__":
    main()