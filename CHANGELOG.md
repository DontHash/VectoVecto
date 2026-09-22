# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[semantic](https://semver.org/).

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
