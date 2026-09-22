# tier_c photo fine-tune — acceptance results (parked)

Run: RealESRGAN-style RRDBNet fine-tune from `RealESRGAN_x4plus.pth`, DIV2K train,
L4 on-demand (`sr-train`, us-central1-a). Stopped deliberately at ~9,450/30,000 iters
on 2026-09-20 — val curves peaked at ~1k iters and degraded monotonically afterward.

## Acceptance eval — DIV2K valid, 20 imgs @ 512px, severity=medium

| model    | PSNR   | SSIM   | LPIPS  | DISTS  | clay   | hf    | s/img |
|----------|--------|--------|--------|--------|--------|-------|-------|
| **x4plus (incumbent)** | 20.882 | 0.5321 | **0.3504** | **0.2215** | **0.4149** | **0.573** | 0.3 |
| x4v3     | 21.567 | 0.5649 | 0.4838 | 0.2861 | 0.7795 | 0.078 | 0.0 |
| tier_c g_ema | 22.080 | 0.5495 | 0.3970 | 0.3027 | 0.4935 | 0.488 | 0.3 |
| bicubic  | 21.824 | 0.5417 | 0.7245 | 0.3482 | 0.9159 | 0.013 | 0.0 |
| lanczos  | 21.808 | 0.5394 | 0.7258 | 0.3490 | 0.9165 | 0.011 | 0.0 |

## Verdict: REJECTED

tier_c wins only on PSNR (distortion), loses every perceptual metric that matters
(LPIPS, DISTS, clay, hf). Full 30k run not worth ~$3: the trajectory was already
worse than x4plus at every checkpoint, so more iterations cannot fix a bad objective
— the loss mix (pixel + percep + gan at this weight) simply favors smoothness.

Consequence: `sr_engine.load_engine("auto")` ignores `artifacts/tier_c/*` unless
`artifacts/tier_c/ACCEPTED` exists (see `tier_c_accepted()` in `sr_engine.py`).
That marker is intentionally absent; x4plus remains the shipped photo default.

Checkpoints kept locally under `artifacts/tier_c/` (gitignored) for a possible
future attempt with a corrected objective (e.g. USM/high-frequency-aware loss,
trained on degraded-document/photo domain rather than DIV2K).

Reproduce:
    python eval_harness_v2.py --hr-dir data/DIV2K_valid_HR --images 20 --severity medium \
      --models "bicubic,lanczos,x4plus,x4v3,pth:artifacts/tier_c/g_ema.pth" \
      --json out/acceptance_medium.json --csv out/acceptance_medium.csv
