# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[semantic](https://semver.org/).

## [Unreleased]

### Added

- **W-A: Nepali lexicon fetch** (`scripts/fetch_nepali_lexicon.py`): merges
  tesseract-ocr/langdata `nep.wordlist` (Apache-2.0) and
  `nepali-brihat-sabdakosh-json` (MIT) into a 132,997-word NFC-normalized list
  plus a sha256 manifest under `data/lexicon/` (git-ignored; not
  redistributed). Air-gapped builds pass local sources.
- **W-B: `unknown_word` review flag** (opt-in): flags Devanagari tokens whose
  words are all out-of-lexicon; digit-bearing tokens are skipped and text is
  never changed. Frozen confirmation: modern-PDF token queue R@10
  0.148 → **0.175**, P@10 +0.4pp, digit queue unchanged; letterpress neutral.
  Active only when the lexicon is present (`VECTOVECTO_LEXICON` overrides).
  Dev-tuning tool: `scripts/tune_lexicon_flag.py`; decision record:
  [docs/PLAN.md](docs/PLAN.md) Appendix T; gate row in
  [docs/EVALUATION.md](docs/EVALUATION.md).
- **W-D: real-print textbook probe** (`scripts/harvest_nepali_textbooks.py`,
  `scripts/profile_textbook_scans.py`): CDC catalogue downloads are dead
  (404 with and without session) and the MOEST eLibrary text layers are
  legacy-font mojibake (20/20 gated out), so no born-digital textbook GT set
  exists; 4 Cornell textbook scans (collection 1813/24179, no text layers)
  are profiled as an unlabeled probe — 26.3 tokens/page, 24.8% flagged, 4.0%
  `unknown_word`, 1.46 s/page median, no CER claims. Negative results and
  sources recorded in [docs/PLAN.md](docs/PLAN.md) Appendix U.
- **W-C: CC-100 corpus text experiment** (`scripts/fetch_deva_corpus.py`,
  `--corpus` in `scripts/export_training_data.py`): 100k filtered Nepali
  sentences replace the 30% word-pool branch of the synthetic mix (digit
  branches unchanged); the corpus-trained model passes the frozen gate
  (digit-exact **0.815** [0.775, 0.853] vs 0.810 [0.769, 0.847]) but every
  delta is inside the frozen-set resolution, so the shipped v8 weights stay
  ([docs/PLAN.md](docs/PLAN.md) Appendix W). Also fixes the GCS output path in
  `deva_crnn.train` (checkpoints were written to a literal `gs:/...` dir and
  never uploaded) and adds `scripts/gcp_w1_vm_startup.sh` with the DLVM
  proprietary-driver guard for GCP training.
- **W1: Devanagari searchable PDF fixed**: the invisible text layer and the
  overlay annotations now use the bundled Mukta (OFL-1.1, `fonts/`) instead of
  Helvetica/Hershey, so Nepali/Hindi pages extract as real searchable text and
  Devanagari `alt_text` renders in the review overlay. Fallback chain: bundled
  font → system fonts (Mangal/Lohit/Noto) → Helvetica with a logged warning.
  Noto Sans Devanagari was measured to lose ASCII letters in reportlab's
  subsetter, hence Mukta; extraction round-trips are pinned by tests.
- **W5: logging foundation**: `logging_setup.py` configures a `vectovecto.*`
  stderr logger (idempotent, `VECTOVECTO_LOG_LEVEL`, default INFO) from both
  entry points. Server-side pipeline failures, unreadable inputs, CLI
  per-file errors, previously swallowed audit/verifier errors and the photo
  bicubic fallback are now logged instead of only printed or hidden; the
  Gradio status output is unchanged.
- **Phase 1 VLM bake-off (Qwen3-VL)**: `qwen3vl` (fp16 8B), `qwen3vl-4b`,
  `qwen3vl-8b-4bit` and `qwen3vl-8b-4bit-rt` candidates in
  `evals/harness/bakeoff_models.py`, with a deterministic 1 Mpx pixel cap and
  a VRAM reservation for offload headroom; `--lines-source deva_real_lines`
  and median-CER + catastrophic-rate stats in the bake-off summaries. Runs:
  SPOT V100 fp16 (`scripts/gcp_vlm_8b_startup.sh`) and Kaggle T4 runtime-NF4
  (`kaggle/vlm_eval/`). Frozen result: the 8B cuts page CER 0.53 → **0.176**
  on the hardest 10 modern pages (reading order) but costs 152.7 s/page and
  invents 224 tokens; lines: median CER 0.0 yet digit-exact 0.077 (Bengali
  numerals) — **no VLM mode ships**; evidence in
  [evals/bakeoff_results.md](evals/bakeoff_results.md) and
  [docs/PLAN.md](docs/PLAN.md) Appendix X.

### Fixed

- **`doc_data.load_dataset` on POSIX**: in-dir manifests written on Windows
  stored `pages\x.png`; joining on Linux produced literal-backslash paths, so
  every page silently scored zero (GCP pages bake-off). Relative paths are now
  separator-normalized and the frozen `{"path": ...}` form is accepted;
  regression test in `tests/test_doc_data_mixed.py`.

## [1.2.0] - 2026-09-22

### Added

- **Devanagari line reader (opt-in)**: `--deva-lines on` runs the trained
  CRNN+CTC recognizer on RapidOCR's line boxes. Frozen-gate digit-exact
  **0.810 [0.769, 0.847]** (bar 0.72); inside the pipeline on frozen letterpress
  pages: page CER **0.434 -> 0.253**, bagCER 0.553 -> 0.434, +0.8 s/page,
  review queue 350 flags smaller, no silent invented digits. It is off by
  default because it measurably hurts modern table pages (+18pp CER); PDF
  inputs never use it. Weights stay server-side (`VECTOVECTO_DEVA_CKPT` or
  `weights/deva_crnn_h48w512.pt`); the recipe is in `docs/TRAINING.md`.
- `digit_added` flag: any number the reader saw that the engine did not is
  flagged for review ? never silent.
- Kaggle GPU training toolchain (`kaggle/w1_train/`, config-driven private
  kernel) and GCS-aware trainer options (`--in-h/--in-w`, cosine schedule,
  `--save-best`).

### Changed

- Gate reporting now includes bootstrap 95% CIs.

### Fixed

- Line-box merging for the reader: adjacent detector fragments merge into one
  line (the recognizer is line-level) while a drawn table rule blocks the
  merge; born-digital PDFs keep the engine reading.

## [1.1.0] - 2026-09-22

The "ship it honestly" release: measured queue wins, multi-page PDF, and a
product-first repository.

### Added

- **Multi-page PDF**: `--max-pages 0` processes every page and writes
  `<stem>_combined.pdf` / `.txt`; the studio gained an "All pages (PDF)"
  checkbox (default remains the first page).
- **2× digit re-pass, default ON for Devanagari**: frozen review-queue digit
  recall@10 **0.73 → 0.88** (letterpress) and **0.15 → 0.38** (government
  PDFs) at ≈ +0.4 s/page. Latin pages keep it off.
- **`invalid_sequence` flag** (impossible combining sequence): token R@10
  +3.0pp letterpress / +8.0pp PDFs on the frozen sets.
- **Optional digit verifier** (`--digit-verifier bodhan`) with a junk-crop
  filter and 64-token cap (32 crops: 192 s → 26 s); stays opt-in because the
  cost gate failed.
- **Hosted web app path**: `Dockerfile`, env-configured host/port, optional
  basic auth, `VECTOVECTO_DOCUMENT_ONLY` to hide the non-commercial photo tab.
- **Repository/product structure**: `docs/` (architecture, evaluation,
  deployment, licensing, release, plan), `evals/harness/` for evaluation
  tooling, `scripts/` for data/training/release tools, `legacy/` for archived
  research, `pyproject.toml` with entry points, `CHANGELOG.md`, license gate.
- **Evaluation evidence**: frozen manifests + harnesses for letterpress,
  government PDFs, real line crops and a photo proxy; Gemini-2.5-Pro anchor
  workflow with a human review worksheet.
- **Devanagari line recognizer research track** (`deva_crnn/`, not adopted):
  multi-font synthetic data, scan-realistic augmentation, montage labeling of
  real lines, warm-start training, and a pre-registered adopt-if gate.

### Changed

- Brand unified to **VectoVecto** (README, studio title, package metadata).
- `--repass-digits/--no-repass-digits` documented as the Devanagari default.
- README rewritten product-first; every claim links to a frozen-set number.

### Fixed

- Pipeline coordinate space is fixed at entry (`fit_max_side`) and recorded in
  metadata, so restore/resize never shifts token boxes.
- Verifier junk-crop filter prevents watermark/URL strips from burning seconds
  per page.

### Removed

- Pre-pivot research assets from the repository tree (photo-training notebook,
  cloud-VM training scripts moved to `legacy/`, demo media, superseded roadmap).

### Known limitations

- Devanagari line recognizer attempts failed the gate (best digit-exact 0.424
  vs bar 0.75) — RapidOCR remains the engine.
- Real phone-photo field set still pending; numbers come from a labeled proxy.
- Letterpress preprocessing (sauvola) rejected: invents tokens.

## [1.0.0] - 2026-08

Initial product: local document restore with searchable PDF, overlay,
transcript and OCR JSON; Devanagari support; orientation/layout/reading-order
handling; risk-flag queue; Gradio studio + CLI; photo upscaler sharing the
same entry points.
