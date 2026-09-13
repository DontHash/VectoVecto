"""
eval_deep_sr.py — Tier B quantitative evaluation (PSNR / SSIM).

Roadmap Phase 0 demands numbers, not vibes: every method is scored on held-out
DIV2K validation images against ground truth. This script answers "did Tier B
actually beat bicubic?" with a table instead of a contact sheet.

Protocol per image:
  1. Center-crop HR to --hr-size (default 512; 0 = full image).
  2. Degrade HR ONCE with degrade_known_kernel (the same Real-ESRGAN-style
     pipeline used in training) -> (LR, sigma, kernel).
  3. Every method upscales that SAME LR, so the comparison is identical input.
  4. PSNR/SSIM (RGB, data_range=255) against HR, using eval_harness metrics.

Methods compared:
  - Bicubic (torch F.interpolate)
  - Lanczos (cv2 INTER_LANCZOS4)
  - Tier B: Deep Unfolding SR (DRUNet) from --ckpt

Usage:
    python eval_deep_sr.py                       # DIV2K valid, 20 x 512px crops
    python eval_deep_sr.py --num-images 5 --device cpu
    python eval_deep_sr.py --hr-size 0 --num-images 10   # full-resolution images
    python eval_deep_sr.py --save-crops eval_out --json eval_out/metrics.json
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from drunet import DRUNet
from deep_unfolding import DeepUnfoldingSR
from degradation import degrade_known_kernel
from eval_harness import calculate_psnr, calculate_ssim

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "artifacts", "deep_sr", "best_checkpoint.pth")
LOCAL_VALID_HR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "DIV2K", "DIV2K_valid_HR")


def _auto_detect_valid_hr():
    """
    Locate DIV2K validation HR images across environments:
      1. Env override DIV2K_VALID_HR_DIR
      2. Kaggle dataset mount: /kaggle/input/*/.../DIV2K_valid_HR
      3. Local prepare_div2k.py output: ./data/DIV2K/DIV2K_valid_HR
    Returns (dir, is_validation_split) or (None, None).
    """
    env_dir = os.environ.get("DIV2K_VALID_HR_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir, True

    if os.path.isdir("/kaggle/input"):
        for entry in sorted(os.listdir("/kaggle/input")):
            base = os.path.join("/kaggle/input", entry)
            if not os.path.isdir(base):
                continue
            for sub in ("DIV2K_valid_HR", "DIV2K_valid_HR/DIV2K_valid_HR", "valid_HR"):
                p = os.path.join(base, sub)
                if os.path.isdir(p) and _has_images(p):
                    return p, True

    if os.path.isdir(LOCAL_VALID_HR) and _has_images(LOCAL_VALID_HR):
        return LOCAL_VALID_HR, True

    # Last resort: training split (contaminated - scores will be optimistic).
    train = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "DIV2K", "DIV2K_train_HR")
    if os.path.isdir(train) and _has_images(train):
        return train, False

    return None, None


def _has_images(d):
    return any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(d))


def _to_uint8(tensor_chw):
    """(3,H,W) float [0,1] -> (H,W,3) uint8."""
    arr = tensor_chw.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)


def _center_crop(img, size):
    """Center-crop a HxWx3 uint8 image to size x size (or keep whole image)."""
    h, w = img.shape[:2]
    size = min(size, h, w)
    y = (h - size) // 2
    x = (w - size) // 2
    return img[y:y + size, x:x + size]


def _load_generator(ckpt_path, scale, fallback_iters, device):
    """Build DeepUnfoldingSR(DRUNet) and load weights. Returns (model, iters)."""
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Train first (train_deep_sr.py) or download best_checkpoint.pth "
              "from Kaggle into artifacts/deep_sr/.", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    iters = state["alphas"].numel() if "alphas" in state else fallback_iters

    model = DeepUnfoldingSR(DRUNet(in_channels=3, num_feat=64, num_blocks=20),
                            iterations=iters, scale=scale).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, iters


def evaluate(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}")

    hr_dir = args.hr_dir
    is_valid_split = True
    if hr_dir is None:
        hr_dir, is_valid_split = _auto_detect_valid_hr()
    if hr_dir is None:
        print("ERROR: DIV2K HR images not found.", file=sys.stderr)
        print("Options: --hr-dir PATH, set DIV2K_VALID_HR_DIR, or run "
              "`python prepare_div2k.py --val` locally.", file=sys.stderr)
        sys.exit(1)
    if not is_valid_split:
        print("WARNING: validation split not found - evaluating on the TRAINING "
              "split. These images were used for training, so scores are optimistic.",
              file=sys.stderr)
    print(f"HR images: {hr_dir}")

    model, iters = _load_generator(args.ckpt, args.scale, args.unfolding_iters, device)
    print(f"Checkpoint: {args.ckpt} (unfolding K={iters}, scale={args.scale}x)")

    files = sorted(f for f in os.listdir(hr_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        print("ERROR: no images found in HR dir.", file=sys.stderr)
        sys.exit(1)
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[:args.num_images]
    print(f"Evaluating {len(files)} image(s) | HR crop: "
          f"{'full' if args.hr_size == 0 else str(args.hr_size) + 'px'}\n")

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    results = {"bicubic": [], "lanczos": [], "tier_b": []}
    per_image = []

    for idx, name in enumerate(files, 1):
        img_bgr = cv2.imread(os.path.join(hr_dir, name), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  SKIP {name} (unreadable)")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if args.hr_size > 0:
            img = _center_crop(img, args.hr_size)

        hr_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        # One degradation per image; all methods see the same LR.
        lr_t, sigma, kernel = degrade_known_kernel(hr_t, scale=args.scale)
        lr_t = lr_t.to(device)
        kernel = kernel.to(device)
        sigma_t = torch.tensor(float(sigma), device=device)

        # --- Bicubic ---
        sr_bicubic = F.interpolate(lr_t, scale_factor=args.scale, mode="bicubic",
                                   align_corners=False)
        # --- Lanczos (cv2 works on uint8) ---
        lr_u8 = _to_uint8(lr_t[0])
        h_lr, w_lr = lr_u8.shape[:2]
        sr_lanczos = cv2.resize(lr_u8, (w_lr * args.scale, h_lr * args.scale),
                                interpolation=cv2.INTER_LANCZOS4)
        # --- Tier B ---
        with torch.no_grad():
            sr_tier_b = model(lr_t, kernel, sigma_noise=sigma_t)

        # Align every output to HR (degradation may drop uneven border pixels).
        min_h = min(sr_tier_b.shape[-2], sr_bicubic.shape[-2], img.shape[0])
        min_w = min(sr_tier_b.shape[-1], sr_bicubic.shape[-1], img.shape[1])
        hr_u8 = img[:min_h, :min_w]
        out = {
            "bicubic": _to_uint8(sr_bicubic[0, :, :min_h, :min_w]),
            "lanczos": sr_lanczos[:min_h, :min_w],
            "tier_b": _to_uint8(sr_tier_b[0, :, :min_h, :min_w]),
        }

        row = {"image": name, "sigma": round(float(sigma), 4)}
        for method, arr in out.items():
            p = calculate_psnr(hr_u8, arr)
            s = calculate_ssim(hr_u8, arr)
            results[method].append((p, s))
            row[method] = {"psnr": round(float(p), 3), "ssim": round(float(s), 5)}

        per_image.append(row)
        print(f"  [{idx}/{len(files)}] {name}: "
              f"bicubic {row['bicubic']['psnr']:.2f}/{row['bicubic']['ssim']:.4f} | "
              f"lanczos {row['lanczos']['psnr']:.2f}/{row['lanczos']['ssim']:.4f} | "
              f"tier_b {row['tier_b']['psnr']:.2f}/{row['tier_b']['ssim']:.4f}")

        if args.save_crops:
            base = os.path.join(args.save_crops, os.path.splitext(name)[0])
            cv2.imwrite(base + "_hr.png", cv2.cvtColor(hr_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(base + "_lanczos.png", cv2.cvtColor(out["lanczos"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(base + "_bicubic.png", cv2.cvtColor(out["bicubic"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(base + "_tier_b.png", cv2.cvtColor(out["tier_b"], cv2.COLOR_RGB2BGR))

    if not per_image:
        print("ERROR: no images evaluated.", file=sys.stderr)
        sys.exit(1)

    # --- Summary table ---
    def mean(method):
        arr = np.array(results[method], dtype=np.float64)
        return arr[:, 0].mean(), arr[:, 1].mean()

    print("\n=== Mean over {} image(s) (RGB PSNR / SSIM) ===".format(len(per_image)))
    print(f"{'method':<12} {'PSNR (dB)':>10} {'SSIM':>10}")
    summary = {}
    for method in ("bicubic", "lanczos", "tier_b"):
        p, s = mean(method)
        summary[method] = {"psnr": round(float(p), 3), "ssim": round(float(s), 5)}
        print(f"{method:<12} {p:>10.2f} {s:>10.4f}")

    delta = summary["tier_b"]["psnr"] - summary["bicubic"]["psnr"]
    print(f"\nTier B vs bicubic: {delta:+.2f} dB PSNR, "
          f"{summary['tier_b']['ssim'] - summary['bicubic']['ssim']:+.4f} SSIM")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "mean": summary, "per_image": per_image},
                      f, indent=2)
        print(f"Metrics written to: {args.json}")


def parse_args():
    p = argparse.ArgumentParser(description="Tier B evaluation: PSNR/SSIM vs baselines")
    p.add_argument("--hr-dir", default=None,
                   help="HR image folder. Auto-detected from Kaggle / DIV2K_VALID_HR_DIR / "
                        "./data/DIV2K/DIV2K_valid_HR if omitted.")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Tier B checkpoint path")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--hr-size", type=int, default=512,
                   help="Center-crop HR to NxN before degrading. 0 = full image.")
    p.add_argument("--unfolding-iters", type=int, default=5,
                   help="Fallback K if the checkpoint doesn't carry alphas")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-crops", default=None,
                   help="Directory to write HR/bicubic/lanczos/tier_b crops for eyeballing")
    p.add_argument("--json", default=None, help="Write full metrics to this JSON path")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
