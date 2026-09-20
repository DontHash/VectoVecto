# VectorScaling — Document Restore (+ photo upscaler)

Local, offline document restoration: a photo or scan goes in, a cleaned page
with a **searchable PDF**, overlay, transcript and OCR JSON comes out — with
honest flags for the numbers it is unsure about ("won't invent the numbers on
your bill"). A photo upscaler (Real-ESRGAN x4plus) shares the same CLI/app.

Everything runs on your machine. No cloud calls, no telemetry.

## Install

```bash
pip install -r requirements.txt
```

Optional:

- [Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki) — second OCR
  backend (demoted to clean-scan/synthetic fallback) and the OSD rotation
  fallback.
- Dev/eval tooling (SROIE/CORD/arXiv eval sets, metrics): `pip install -r requirements-dev.txt`

## Usage

### CLI

```bash
# Document mode: photo/scan in, searchable PDF + overlay + txt + JSON out
python cli.py --mode document --input page.jpg --output out/run1

# Multi-page PDF (page 1 by default; --max-pages N to extend)
python cli.py --mode document --input scan.pdf --output out/run1 --max-pages 3

# Knobs (all default to the measured best config)
#   --ocr rapidocr|tesseract   --lang <code>      --deskew
#   --no-reading-order         --rotate auto|off --repass-digits
#   --no-pdf --no-overlay --no-txt
```

Outputs per page: `*_restored.png` (or `.jpg`/`.webp`), `*_searchable.pdf`,
`*_overlay.png` (numbered review boxes), `*_transcript.txt`, `*_ocr.json`
(tokens, flags, review queue, orientation/reading-order evidence).

### App

```bash
python app.py            # Gradio studio, document tab first
```

### Photo mode

```bash
python cli.py --mode photo --input photos/ --output out/upscaled --scale 4
```

## What it does (and what it deliberately does not)

Measured behaviour, not marketing (full tables in
[`document_restore_plan.md`](document_restore_plan.md)):

| Capability | Status |
|---|---|
| Orientation (EXIF + 0/90/180/270 via OCR evidence) | syn 16/16; real upright 0/30 false rotations; rotated 90/90 decided |
| Single-column layout | identity — 60/60 real pages untouched |
| Two-column reading order | real arXiv set: WER 0.870→0.268 (−69%) on split pages, CER 0.607→0.241 end-to-end |
| Digit-number honesty | review queue ranked by risk; best case recalls ~23% of digit errors in the top-5 — **weak, documented, improving** |
| Tesseract backend | fails on real photos (CORD CER 0.90 raw / 1.57 restored) — clean-scan fallback only |
| Numeric flags bar (coverage ≥0.55) | **not met** on real photos; replaced by the ranked review queue |
| 180° via soft evidence only | refused, flagged `orientation?` instead of a blind flip |
| High-Fidelity photo model | rejected (perceptual metrics) and gated behind `artifacts/tier_c/ACCEPTED` |
| License | RapidOCR/PP-OCR Apache-2.0; reportlab/pypdfium2 permissive; PyMuPDF (AGPL) unused; UltraSharp weights CC-BY-NC-SA (not for commercial builds) |

## Tests

```bash
python -m pytest tests/ -q
```

77 tests: OCR/export/routing/CLI/app, layout (13), orientation (13),
engine state regressions. `legacy/` holds archived photo-training scripts and
is not part of the product.
