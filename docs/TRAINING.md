# Training the Devanagari line reader

The shipped Devanagari recognizer (W1) is a small CRNN+CTC trained in this
repository. The **weights are not part of the repository** — deployments
provide them via `VECTOVECTO_DEVA_CKPT` or `weights/deva_crnn_h48w512.pt`.
This document is the reproducible recipe, with the measured results that
justify each choice.

## What it is

- Input: a grayscale line crop normalized to **48 x 512** (height x width,
  right-padded). Width matters more than height: real letterpress lines are
  ~1,432 px wide and W=256 squeezed them 5.6x; W=512 halves that and doubles
  the CTC timesteps (T=128).
- Architecture: 5 conv blocks (GroupNorm — BatchNorm diverged at small
  batches) + 2-layer BiLSTM (hidden 256) + CTC head.
- Decoder: greedy. CTC prefix beam search was implemented, verified against
  brute force, and **rejected**: identical digit-exact, 20x slower.

## Data

| source | lines | notes |
|---|---|---|
| synthetic (local Qt rendering) | 36,000 | multi-font (Nirmala + Adobe Devanagari regular/bold/italic), per-line letter-spacing/stretch jitter, 30% short table cells, 25% long lines (40-60 chars), punctuation parity with the real corpus, heavy scan augmentation (ink spread, resolution loss, illumination, blur, noise, JPEG, rotation) |
| heiDATA train books | 1,898 | CC BY 4.0, doi:10.11588/data/EGOKEI; 6 books not used by the frozen sets |
| heiDATA dev books | 1,322 | same collection; consumed as training data (no longer a tuning set) |
| Gemini-labeled modern PDF lines | 745 | `scripts/label_real_lines.py` montage labeling of extra pages outside the frozen anchor |

Real lines are repeated x4 and digit-bearing lines x2 (the gate metric is
digit-exact). The frozen sets (`evals/manifests/`) are **never** used for
training or tuning.

## Reproduce

```bash
# 1. Data: download the unused heiDATA books, split train/holdout, export the npz
python scripts/harvest_heidata_books.py
python scripts/build_heidata_training_sets.py --freeze
python scripts/export_training_data.py \
  --out deva_crnn/data/train_v8_h48w512.npz --n 36000 --height 48 --width 512 \
  --fonts all --aug-level heavy --cell-frac 0.3 \
  --real-heidata data/doc_eval/heidata_train \
  --real-heidata data/doc_eval/heidata_dev \
  --real-dir data/doc_eval/deva_real_lines --real-repeat 4 --digit-repeat 2

# 2. Train (Kaggle GPU, config-driven kernel in kaggle/w1_train/)
#    w1_config.json selects the npz/height/width/epochs; push with:
kaggle datasets version -p out/kaggle_w1 -m "..."      # private dataset
kaggle kernels push -p kaggle/w1_train                  # private GPU kernel

# or locally (CPU/GPU), same config:
python -m deva_crnn.train --data deva_crnn/data/train_v8_h48w512.npz \
  --out out/deva_crnn --epochs 26 --batch 64 --lr 1e-3 --in-h 48 --in-w 512

# 3. Gate (pre-registered bar: digit-exact >= 0.72 on frozen heiDATA lines)
python scripts/eval_deva_crnn_gate.py --ckpt out/deva_crnn/ckpt.pt \
  --json out/deva_crnn_gate.json

# 4. Integration gate (the reader inside the pipeline)
python scripts/eval_deva_lines_integration.py \
  --data-dir data/doc_eval/heidata_printed \
  --frozen evals/manifests/heidata_printed_v1.json --lang ne
```

## Measured results (frozen sets)

| model | digit-exact (heiDATA lines) | 95% CI | CER | bagCER | holdout |
|---|---|---|---|---|---|
| h=32, W=256 | 0.603 | - | 0.341 | 0.451 | 0.483 |
| h=48, W=256 | 0.710 | - | 0.282 | 0.384 | 0.691 |
| **h=48, W=512 (shipped)** | **0.810** | [0.769, 0.847] | **0.177** | **0.235** | **0.754** |

Integration (RapidOCR detection + reader recognition, frozen letterpress
pages): page CER 0.434 -> **0.253**, bagCER 0.553 -> **0.434**, +0.77 s/page,
review queue 350 flags smaller, no silent invented digits.

**Scope (measured):** the reader is a *line* model for running text. It helps
letterpress book scans and hurts modern table pages (+18pp CER on born-digital
registers, +22pp on their degraded photo proxies) — so it ships **opt-in**
(`--deva-lines on`, default `off`), and the CLI keeps it off for PDF inputs.

## Provenance and licensing

- heiDATA "Ground Truth data for printed Devanagari" — CC BY 4.0,
  doi:10.11588/data/EGOKEI. Model weights trained on it inherit the
  attribution requirement (see [LICENSES.md](LICENSES.md)).
- Modern-PDF line crops derive from Nepali government publications (internal
  evaluation use only; the derived npz datasets on Kaggle are private).
- Synthetic data is generated locally by `doc_data.py` (Qt-shaped rendering).
