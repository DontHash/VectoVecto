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
#   --ocr rapidocr|tesseract   --lang en|ne|hi     --deskew
#   --no-reading-order         --rotate auto|off --repass-digits
#   --digit-verifier off|bodhan  (optional second-model digit check)
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
| Orientation (EXIF + 0/90/180/270 via OCR evidence) | syn 16/16; real upright 0/30 false rotations (and 0/19 on unlabeled Nepali scans); rotated 90/90 decided |
| Devanagari (Nepali/Hindi) — rendered fixture | clean CER 0.030/0.028 (Latin engine 0.85 — the language model is essential); degraded ne 0.094 (pass) / hi 0.170 (heavy fails) |
| Devanagari — real pages (**the honest bar**) | letterpress ALTO-verified (n=69): CER 0.434 [0.387–0.481], exact-token recall 0.647 [0.619–0.675]. Modern gov PDFs (n=41, **Gemini 2.5 Pro GT anchor**, bodhan-corroborated 0.881): CER **0.135 [0.097–0.186]** / bagCER 0.142, exact-token recall 0.877 [0.732–0.968]; per-doc bagCER 0.06–0.20. The older 0.338 was scored against a corrupt text layer (41/41 pages >2% invalid tokens) and is historical only |
| Devanagari — alternative models (bake-off, Appendix L) | Tesseract/TrOCR/GLM-OCR all scored on frozen sets and **rejected**: invented tokens, domain mismatch, or 200 s/page. The harness stays for future models (`python eval_models.py --list`) |
| Devanagari — real line crops (n=500) | CER 0.717 [0.693–0.740]; GT audit: 0.07% invalid tokens — the defect is *alignment* (machine-generated GT, partly misaligned), so numbers stay "behavior only" until a human pass |
| Detection on real pages | covers 97.5% of ALTO line boxes (area view); not the bottleneck — recognition is |
| Single-column layout | identity — 60/60 real pages untouched |
| Two-column reading order | real arXiv set: WER 0.870→0.268 (−69%) on split pages, CER 0.607→0.241 end-to-end; all 4 unlabeled real splits verified by geometry; >2 columns unsupported |
| Digit-number honesty | review queue ranked by risk; best case recalls ~23% of digit errors in the top-5 — **weak, documented**; on real Devanagari pages flag coverage is 0.2–1.6% (calibration ECE ≈ 0.82) |
| Devanagari flags + calibration (Appendix M) | `script_mismatch` + conf-90 digit bar + isotonic `cal_conf`: on letterpress scans the queue surfaces **75% of digit errors in the top 10** (with the optional verifier; P@10 0.16); on clean gov PDFs it barely helps (R@10 0.15) — errors there are confidently wrong. Calibration is monotone (ranking unchanged); ECE 0.82→0.30 on scans |
| Optional digit verifier (`--digit-verifier bodhan`, Appendix L) | second model re-reads suspect digits; disagreement is flagged `cross_model_conflict` with the alt reading kept (text never changed). Frozen: flag recall 0.71 / precision 0.81; queue R@10 0.730→0.752 (`--digit-verifier-scope flagged`). Scope `all` gains +3.7pp R@10 but costs +7.4 s/page — pre-registered gate failed, stays opt-in (Appendix N). Needs a one-time `hf auth login` + license acceptance, ~1.9 GB. License: Indic Open Model License 1.0 (self-host OK, no third-party hosting, attribution) |
| Multi-page PDF (P6) | `--max-pages 0` processes every page and writes `<stem>_combined.pdf` / `.txt` (per-page artifacts unchanged); GUI "All pages (PDF)" checkbox, default stays first page. Verified: 3-page demo → 3-page searchable PDF, text extractable on every page |
| Mixed-page router (P5, opt-in `--mixed-router`) | restored text regions + untouched logos/photos/signatures (non-text PSNR 60–67 dB vs 18–23 dB after a naive full-page restore). Pre-registered gates: text CER +0%, cost 0.31 s/page, non-text PASS, but 1 invented token on a photo texture → **not auto-enabled** (Appendix O) |
| Tesseract backend | fails on real photos (CORD CER 0.90 raw / 1.57 restored) — clean-scan fallback only |
| Numeric flags bar (coverage ≥0.55) | **not met** on real photos; replaced by the ranked review queue |
| Frozen eval + CIs | every real-set number above comes from a hash-frozen manifest (`evals/manifests/`) with bootstrap 95% CIs; see plan Appendix K |
| 180° via soft evidence only | refused, flagged `orientation?` instead of a blind flip |
| High-Fidelity photo model | rejected (perceptual metrics) and gated behind `artifacts/tier_c/ACCEPTED` |
| License | RapidOCR/PP-OCR Apache-2.0; reportlab/pypdfium2 permissive; PyMuPDF (AGPL) unused; UltraSharp weights CC-BY-NC-SA (not for commercial builds); eval sets internal-only unless stated |

## Tests

```bash
python -m pytest tests/ -q
```

189 tests: OCR/export/routing/CLI/app (incl. multi-page combined PDF and the
mixed-page router), layout (14), orientation (13), language plumbing,
engine-state regressions, frozen-manifest + bootstrap-CI guards, GT-validity
audits, error taxonomy, anchor harness, Unicode-path IO, line eval, box
metrics, sanity audit. `legacy/` holds archived photo-training scripts and is
not part of the product.

## Real-data evaluation (frozen)

Real Devanagari evidence lives under `evals/manifests/` (content-hash frozen):

```bash
python eval_freeze.py --check evals/manifests/heidata_printed_v1.json
python eval_document.py --data-dir data/doc_eval/heidata_printed --frozen \
    evals/manifests/heidata_printed_v1.json --lang ne --methods raw --bootstrap 2000
python eval_lines.py --data-dir data/doc_eval/nepali_lines --frozen \
    evals/manifests/nepali_lines_v1.json --lang ne --bootstrap 2000
```

Rebuild instructions and dataset provenance are recorded per manifest and in
plan Appendix K. Frozen sets are never used to tune thresholds; a dataset
change is a new freeze version.

Ground-truth integrity is checked before freezing: `doc_metrics.devanagari_validity()`
reports the invalid-combining-sequence rate of Devanagari text (reordered
matras, dangling viramas, orphan marks), and `harvest_nepali_pdfs.py` rejects
PDFs whose text layer scores >2%. Where no clean reference exists, the anchor
is model-produced and labeled: `anchor_gemini.py` (Vertex AI) transcribes the
pages, the local engines corroborate (`eval_anchor.py --score`), and the
disagreement list is the human review artifact (`out/anchor_gemini/audit.html`).
