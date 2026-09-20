"""
train_v2.py — professional SR fine-tuning (Real-ESRGAN recipe, corrected).

What is different from train_deep_sr.py:
  * Model: RRDBNet x4 (RealESRGAN_x4plus base) or SRVGGNetCompact, NOT the 1.5M
    unfolding denoiser. Capacity was the #1 bottleneck.
  * Degradation: true second-order Real-ESRGAN (degradation_v2), not
    Gaussian-only.
  * Losses: L1 + VGG perceptual + RaGAN (U-Net D) + optional focal-frequency.
    No TV term (a documented cause of the "clay" look).
  * EMA generator, warmup+cosine LR, AMP, grad clipping.
  * Checkpoint "best" selected by validation score (PSNR + 20*SSIM), not by the
    noisy GAN generator loss.
  * Resumable, time-budgeted, GCS-synced: safe on spot instances.

Example (GCP L4 spot):
  python train_v2.py \
    --hr-dirs /opt/data/DIV2K_train_HR \
    --val-hr-dir /opt/data/DIV2K_valid_HR \
    --base-weights weights/RealESRGAN_x4plus.pth \
    --out-dir artifacts/tier_c \
    --iters 30000 --batch 6 --patch 96 --amp bf16 \
    --val-every 1000 --save-every 500 \
    --gcs-bucket gs://theproject-sr-artifacts/tier_c \
    --time-budget-min 380 --shutdown-on-finish
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import degradation_v2 as deg  # noqa: E402
from rrdbnet import RRDBNet  # noqa: E402
from srvggnet import SRVGGNetCompact  # noqa: E402
from unet_discriminator import UNetDiscriminatorSN, RaGANLoss  # noqa: E402
from perceptual_loss import VGGPerceptualLoss  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gcs_upload(local: str, dst: str):
    if not shutil.which("gcloud"):
        return
    try:
        subprocess.Popen(["gcloud", "storage", "cp", local, dst],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[gcs] upload failed: {e}")


def gcs_download(src: str, local: str) -> bool:
    if not shutil.which("gcloud"):
        return False
    try:
        r = subprocess.run(["gcloud", "storage", "cp", src, local],
                           capture_output=True, text=True, timeout=600)
        return r.returncode == 0 and os.path.exists(local)
    except Exception:
        return False


def save_checkpoint(payload: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def psnr_torch(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(max_val ** 2 / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
    return w.to(dtype)


def ssim_torch(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    a, b = a.clamp(0, 1), b.clamp(0, 1)
    c = a.shape[1]
    win = _gaussian_window(11, 1.5, c, a.device, a.dtype)
    pad = 5
    mu_a = F.conv2d(a, win, groups=c, padding=pad)
    mu_b = F.conv2d(b, win, groups=c, padding=pad)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sig_a = F.conv2d(a * a, win, groups=c, padding=pad) - mu_a2
    sig_b = F.conv2d(b * b, win, groups=c, padding=pad) - mu_b2
    sig_ab = F.conv2d(a * b, win, groups=c, padding=pad) - mu_ab
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / \
           ((mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2))
    return float(ssim.mean())


def focal_frequency_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Focal Frequency Loss (Jiang et al. 2021) — directly penalizes missing
    high-frequency bands; helps against over-smoothing."""
    diff = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho")) - \
        torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"))
    amp = torch.abs(diff) ** alpha
    loss = (amp * (torch.abs(diff) ** 2)).sum(dim=-1).mean()
    return loss / (pred.shape[1] * pred.shape[2] * pred.shape[3])


# ---------------------------------------------------------------------------
# Dataset: random HR crops, on-the-fly second-order degradation
# ---------------------------------------------------------------------------

class HrPatchDataset(Dataset):
    def __init__(self, roots: List[str], scale: int, patch: int, seed: int = 0,
                 max_images: int = 0):
        self.paths: List[str] = []
        for root in roots:
            for dirpath, _dirs, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(IMG_EXTS):
                        self.paths.append(os.path.join(dirpath, f))
        self.paths.sort()
        if max_images:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise SystemExit(f"no images found under {roots}")
        self.scale = scale
        self.patch = patch
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        info = get_worker_info()
        seed = self.seed + (info.seed if info is not None else 0) + i
        rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        path = self.paths[i % len(self.paths)]
        if info is not None and info.num_workers > 0:
            path = self.paths[rng.integers(0, len(self.paths))]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        if random.random() < 0.5:
            img = img[:, ::-1]
        if random.random() < 0.5:
            img = img[::-1]

        hr_size = self.patch * self.scale
        h, w = img.shape[:2]
        if h < hr_size or w < hr_size:
            scale = hr_size / min(h, w)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        top = int(rng.integers(0, max(1, h - hr_size + 1)))
        left = int(rng.integers(0, max(1, w - hr_size + 1)))
        hr = img[top:top + hr_size, left:left + hr_size]

        hr_t = torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        hr_t = hr_t.permute(2, 0, 1)
        lr_t, _ = deg.degrade_hr(hr_t.unsqueeze(0), scale=self.scale)
        return lr_t.squeeze(0), hr_t


def worker_init(worker_id: int):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)

    def state_dict(self):
        return self.shadow


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValPair:
    name: str
    lr: torch.Tensor
    hr: torch.Tensor


def build_val_pairs(hr_dir: str, scale: int, size: int, count: int,
                    severity: str, seed: int = 1234) -> List[ValPair]:
    paths: List[str] = []
    for dirpath, _dirs, files in os.walk(hr_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    if count:
        paths = paths[:count]
    pairs: List[ValPair] = []
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h > size and w > size:
            top, left = (h - size) // 2, (w - size) // 2
            img = img[top:top + size, left:left + size]
        hr = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        hr = hr.permute(2, 0, 1)
        lr, _ = deg.degrade_eval(hr.unsqueeze(0), scale=scale, seed=seed + i, severity=severity)
        pairs.append(ValPair(os.path.splitext(os.path.basename(p))[0], lr.squeeze(0), hr))
    if not pairs:
        raise SystemExit(f"no validation images under {hr_dir}")
    return pairs


@torch.no_grad()
def validate(model: torch.nn.Module, pairs: List[ValPair], device: torch.device,
             lpips_fn=None, amp_dtype=None) -> Dict[str, float]:
    model.eval()
    psnrs, ssims, lps = [], [], []
    for vp in pairs:
        x = vp.lr.unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else _nullcontext():
            y = model(x).float()
        y = y.clamp(0, 1)
        hr = vp.hr.unsqueeze(0).to(device)
        psnrs.append(psnr_torch(y, hr))
        ssims.append(ssim_torch(y, hr))
        if lpips_fn is not None:
            with torch.no_grad():
                lps.append(float(lpips_fn(y, hr).mean()))
    model.train()
    out = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
    if lps:
        out["lpips"] = float(np.mean(lps))
    return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def save_val_samples(model: torch.nn.Module, pairs: List[ValPair], device: torch.device,
                     out_dir: str, scale: int, count: int = 3):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for vp in pairs[:count]:
            x = vp.lr.unsqueeze(0).to(device)
            y = model(x).float().clamp(0, 1)
            hr = vp.hr
            lr_up = F.interpolate(vp.lr.unsqueeze(0), scale_factor=scale,
                                  mode="bicubic", align_corners=False).clamp(0, 1)
            row = torch.cat([lr_up[0], y[0].cpu(), hr], dim=2)  # (3, H, 3W)
            arr = (row.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(out_dir, f"{vp.name}_bicubic_model_hr.png"), arr)
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tier-C professional SR fine-tuning")
    ap.add_argument("--hr-dirs", nargs="+", required=True)
    ap.add_argument("--val-hr-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "artifacts", "tier_c"))
    ap.add_argument("--arch", choices=["rrdbnet", "srvgg"], default="rrdbnet")
    ap.add_argument("--base-weights", default=None,
                    help="pretrained generator to fine-tune (e.g. weights/RealESRGAN_x4plus.pth)")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--patch", type=int, default=64, help="LR patch size (HR patch = patch*scale)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--w-pixel", type=float, default=1.0)
    ap.add_argument("--w-percep", type=float, default=1.0)
    ap.add_argument("--w-gan", type=float, default=0.1)
    ap.add_argument("--w-ffl", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-images", type=int, default=8)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--val-severity", choices=["mild", "medium", "heavy"], default="medium")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-samples-every", type=int, default=2000)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--gcs-bucket", default=None,
                    help="e.g. gs://theproject-sr-artifacts/tier_c (checkpoints + logs synced)")
    ap.add_argument("--time-budget-min", type=float, default=0)
    ap.add_argument("--shutdown-on-finish", action="store_true")
    ap.add_argument("--max-images", type=int, default=0)
    return ap


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    if device.type != "cuda":
        amp_dtype = None
    print(f"[train_v2] device={device} amp={args.amp} arch={args.arch} iters={args.iters}")

    # --- models ---
    if args.arch == "rrdbnet":
        g = RRDBNet(3, 3, 64, 23, 32, scale=args.scale)
    else:
        g = SRVGGNetCompact(3, 3, 64, 32, upscale=args.scale)
    if args.base_weights:
        sd = torch.load(args.base_weights, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "params_ema" in sd:
            sd = sd["params_ema"]
        elif isinstance(sd, dict) and "params" in sd:
            sd = sd["params"]
        missing, unexpected = g.load_state_dict(sd, strict=False)
        print(f"[train_v2] base weights loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    g = g.to(device).train()

    d = UNetDiscriminatorSN(3, 64).to(device).train()
    gan = RaGANLoss()
    percep = VGGPerceptualLoss().to(device).eval()

    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr_d, betas=(0.9, 0.99))
    ema = EMA(g, decay=args.ema_decay)

    scaler = torch.amp.GradScaler("cuda") if (amp_dtype == torch.float16) else None

    start_iter, best_score = 0, -1e9
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        g.load_state_dict(ckpt["g"])
        d.load_state_dict(ckpt["d"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        ema.shadow = ckpt["ema"]
        scaler_state = ckpt.get("scaler")
        if scaler is not None and scaler_state:
            scaler.load_state_dict(scaler_state)
        start_iter = ckpt.get("iter", 0)
        best_score = ckpt.get("best_score", -1e9)
        print(f"[train_v2] resumed from {args.resume_from} @ iter {start_iter}")
    elif args.gcs_bucket:
        remote_latest = args.gcs_bucket.rstrip("/") + "/latest.pth"
        local_latest = os.path.join(args.out_dir, "latest.pth")
        if gcs_download(remote_latest, local_latest):
            ckpt = torch.load(local_latest, map_location="cpu", weights_only=False)
            g.load_state_dict(ckpt["g"])
            d.load_state_dict(ckpt["d"])
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
            ema.shadow = ckpt["ema"]
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            start_iter = ckpt.get("iter", 0)
            best_score = ckpt.get("best_score", -1e9)
            print(f"[train_v2] resumed from GCS @ iter {start_iter}")

    # --- data ---
    dataset = HrPatchDataset(args.hr_dirs, args.scale, args.patch, seed=args.seed,
                             max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.num_workers > 0,
                        prefetch_factor=6 if args.num_workers > 0 else None,
                        worker_init_fn=worker_init)
    data_iter = iter(loader)

    val_pairs: List[ValPair] = []
    if args.val_hr_dir:
        val_pairs = build_val_pairs(args.val_hr_dir, args.scale, args.val_size,
                                    args.val_images, args.val_severity)
        print(f"[train_v2] {len(val_pairs)} validation pairs")

    lpips_fn = None
    try:
        import pyiqa
        lpips_fn = pyiqa.create_metric("lpips", device=device)
    except Exception:
        pass

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    t_start = time.time()
    t_last_log = time.time()
    g_loss_ema = None

    def total_lr(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.iters - args.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return max(args.min_lr_ratio, cos)

    for it in range(start_iter, args.iters):
        lr_scale = total_lr(it)
        for pg in opt_g.param_groups:
            pg["lr"] = args.lr * lr_scale
        for pg in opt_d.param_groups:
            pg["lr"] = args.lr_d * lr_scale

        try:
            lr_batch, hr_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            lr_batch, hr_batch = next(data_iter)
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        # ---- generator step ----
        with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else _nullcontext():
            sr = g(lr_batch)
            l_pix = F.l1_loss(sr.float(), hr_batch)
            l_percep = percep(sr.float(), hr_batch)
            pred_real = d(hr_batch)
            pred_fake = d(sr.float())
            l_gan = gan.gen_loss(pred_real, pred_fake)
            l_ffl = focal_frequency_loss(sr.float(), hr_batch) if args.w_ffl > 0 else 0.0
            l_g = (args.w_pixel * l_pix + args.w_percep * l_percep +
                   args.w_gan * l_gan + args.w_ffl * l_ffl)
        opt_g.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(l_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()
        else:
            l_g.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt_g.step()
        ema.update(g)

        # ---- discriminator step ----
        with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else _nullcontext():
            pred_real = d(hr_batch)
            pred_fake = d(sr.detach().float())
            l_d = gan.disc_loss(pred_real, pred_fake)
        opt_d.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(l_d).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
            scaler.step(opt_d)
            scaler.update()
        else:
            l_d.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
            opt_d.step()

        l_val = l_g.item()
        g_loss_ema = l_val if g_loss_ema is None else 0.98 * g_loss_ema + 0.02 * l_val

        if it % 50 == 0:
            now = time.time()
            rate = 50.0 / max(now - t_last_log, 1e-6)
            t_last_log = now
            msg = (f"[{it:>7}/{args.iters}] lr={args.lr * lr_scale:.2e} "
                   f"g={l_val:.4f} (ema {g_loss_ema:.4f}) d={l_d.item():.4f} "
                   f"pix={l_pix.item():.4f} percep={l_percep.item():.4f} "
                   f"gan={l_gan.item():.4f} {rate:.2f} it/s")
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"iter": it, "g": l_val, "g_ema": g_loss_ema,
                                    "d": l_d.item(), "pix": l_pix.item(),
                                    "percep": l_percep.item(), "gan": l_gan.item()}) + "\n")

        # ---- validation ----
        if val_pairs and (it + 1) % args.val_every == 0:
            eval_model = g.__class__(3, 3, 64, 23, 32, scale=args.scale) if args.arch == "rrdbnet" \
                else SRVGGNetCompact(3, 3, 64, 32, upscale=args.scale)
            eval_model = eval_model.to(device)
            ema.copy_to(eval_model)
            t0 = time.time()
            metrics = validate(eval_model, val_pairs, device, lpips_fn, amp_dtype)
            if "lpips" in metrics:
                score = metrics["psnr"] + 20.0 * metrics["ssim"] - 100.0 * metrics["lpips"]
            else:
                score = metrics["psnr"] + 20.0 * metrics["ssim"]
            print(f"[val @ {it+1}] psnr={metrics['psnr']:.3f} ssim={metrics['ssim']:.4f} "
                  f"{'lpips=%.4f ' % metrics['lpips'] if 'lpips' in metrics else ''}"
                  f"score={score:.3f} ({time.time() - t0:.1f}s)", flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"iter": it + 1, "val": metrics, "score": score}) + "\n")
            if args.save_samples_every and (it + 1) % args.save_samples_every == 0:
                save_val_samples(eval_model, val_pairs, device,
                                 os.path.join(args.out_dir, "val_samples"), args.scale)
            if score > best_score:
                best_score = score
                payload = {"iter": it + 1, "g": g.state_dict(), "d": d.state_dict(),
                           "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                           "ema": ema.state_dict(), "best_score": best_score,
                           "metrics": metrics, "args": vars(args)}
                if scaler is not None:
                    payload["scaler"] = scaler.state_dict()
                save_checkpoint(payload, os.path.join(args.out_dir, "best.pth"))
                print(f"[val] new best -> best.pth (score {best_score:.3f})", flush=True)
                if args.gcs_bucket:
                    gcs_upload(os.path.join(args.out_dir, "best.pth"),
                               args.gcs_bucket.rstrip("/") + "/best.pth")

        # ---- checkpoints / budget / stop ----
        if (it + 1) % args.save_every == 0:
            payload = {"iter": it + 1, "g": g.state_dict(), "d": d.state_dict(),
                       "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                       "ema": ema.state_dict(), "best_score": best_score,
                       "args": vars(args)}
            if scaler is not None:
                payload["scaler"] = scaler.state_dict()
            path = os.path.join(args.out_dir, "latest.pth")
            save_checkpoint(payload, path)
            ema_path = os.path.join(args.out_dir, "g_ema.pth")
            ema_payload = {"iter": it + 1, "params": {k: v for k, v in ema.state_dict().items()}}
            save_checkpoint(ema_payload, ema_path)
            print(f"[ckpt] saved @ {it+1}", flush=True)
            if args.gcs_bucket:
                gcs_upload(path, args.gcs_bucket.rstrip("/") + "/latest.pth")
                gcs_upload(ema_path, args.gcs_bucket.rstrip("/") + "/g_ema.pth")
                gcs_upload(log_path, args.gcs_bucket.rstrip("/") + "/train_log.jsonl")

        if args.time_budget_min > 0:
            elapsed_min = (time.time() - t_start) / 60.0
            if elapsed_min >= args.time_budget_min:
                print(f"[train_v2] time budget reached ({elapsed_min:.1f} min) at iter {it+1}",
                      flush=True)
                payload = {"iter": it + 1, "g": g.state_dict(), "d": d.state_dict(),
                           "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                           "ema": ema.state_dict(), "best_score": best_score,
                           "args": vars(args)}
                save_checkpoint(payload, os.path.join(args.out_dir, "latest.pth"))
                break

    # final save
    payload = {"iter": args.iters, "g": g.state_dict(), "d": d.state_dict(),
               "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
               "ema": ema.state_dict(), "best_score": best_score, "args": vars(args)}
    save_checkpoint(payload, os.path.join(args.out_dir, "final.pth"))
    print("[train_v2] done.", flush=True)

    if args.gcs_bucket:
        for f in ("latest.pth", "final.pth", "train_log.jsonl"):
            p = os.path.join(args.out_dir, f)
            if os.path.exists(p):
                gcs_upload(p, args.gcs_bucket.rstrip("/") + "/" + f)

    if args.shutdown_on_finish:
        subprocess.Popen(["shutdown", "-h", "now"])


if __name__ == "__main__":
    main()
