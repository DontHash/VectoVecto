# Architecture

VectoVecto is a local, single-process pipeline. One page in, four artifacts
out; no services, no queue, no network.

```
                       ┌──────────────────────────────────────────────┐
  photo / scan / PDF ─▶│ document_pipeline.run_document_pipeline      │
                       │                                              │
                       │  1 document_orientation  EXIF + 0/90/180/270 │
                       │  2 document_layout       columns, reading    │
                       │                          order               │
                       │  3 document_ocr          RapidOCR/Tesseract  │
                       │                          + digit re-pass      │
                       │  4 document_verifier     optional 2nd model  │
                       │  5 calibration           isotonic conf       │
                       │  6 document_router       text vs non-text    │
                       │                          (opt-in)            │
                       └───────────────┬──────────────────────────────┘
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │ document_export                              │
                       │  searchable PDF · overlay · transcript · JSON│
                       └──────────────────────────────────────────────┘
```

## Module map

### Pipeline (document mode)

| Module | Responsibility |
|---|---|
| `document_pipeline.py` | orchestrates one page (and multi-page runs); resolves the digit re-pass default per language; keeps coordinate space stable across restore/resize |
| `document_ocr.py` | backend registry (`rapidocr`, `tesseract`), token model, risk flags, digit re-pass, verifier hook |
| `document_orientation.py` | EXIF plus OCR-evidence rotation (0/90/180/270); refuses a blind 180 flip, flags `orientation?` instead |
| `document_layout.py` | column detection and reading order (single-column is identity) |
| `document_router.py` | opt-in mixed-page router: restores text regions, leaves logos/photos/signatures untouched |
| `document_restore.py` | restore stream (denoise/contrast) with quality gates |
| `document_verifier.py` | optional second-model digit re-read (`--digit-verifier bodhan`); disagreements become `cross_model_conflict`, text is never silently changed |
| `document_export.py` | searchable PDF (reportlab), numbered review overlay, transcript, OCR JSON; multi-page combined PDF |
| `calibration.py` | isotonic confidence mapping (`cal_conf`) fitted on a dev set |
| `cli.py` | command line: `--mode document|photo`, batch folders, all knobs |
| `app.py` | Gradio studio (document tab first; `VECTOVECTO_DOCUMENT_ONLY=1` hides the photo tab) |

### Data and metrics

| Module | Responsibility |
|---|---|
| `doc_data.py` | dataset IO (`load_dataset`, `imread_safe`), PDF→pages, synthetic fixtures, heiDATA/ALTO ingestion, photo-proxy builder |
| `doc_metrics.py` | CER/WER/bagCER, digit-string metrics, `devanagari_validity`, queue stats, bootstrap CIs |
| `degradation_document.py` | deterministic page degradations (mild/medium/heavy) used by proxies and tests |

### Photo/vector stack (shared CLI/app)

`smart_upscaler.py` (content analysis + routing), `sr_engine.py` (ONNX/torch
model runner), `srvggnet.py` (SRVGG architecture), `tv_refinement.py` (TV
post-processing), `vector_raster_hybrid.py` (raster→vector hybrid output).

### Evidence and research (not part of the shipped runtime)

| Path | Responsibility |
|---|---|
| `evals/harness/` | evaluation harnesses: frozen-set verification, metrics, anchor transcription, model bake-off, gates |
| `evals/manifests/` | content-hash frozen evaluation sets (the numbers in the README come from these) |
| `scripts/` | data acquisition (`harvest_*`), training-data export, real-line labeling, recognizer gate, license/release tooling |
| `deva_crnn/` | Devanagari line recognizer (CRNN+CTC) — trained, gated, **not adopted** yet (see [PLAN.md](PLAN.md) Appendix R2) |
| `legacy/` | archived pre-pivot research (photo SR training, cloud VM scripts) |

## Design rules

1. **Honesty over coverage.** A wrong number is worse than a flagged one: risk
   flags and the review queue are the product, not an afterthought.
2. **Measure, then ship.** Every behaviour claim comes from a frozen set with
   bootstrap CIs; adopt-if gates are written down *before* the experiment.
3. **Offline first.** No runtime network calls; models are vendored by the
   OCR package at install time. Tests enforce this.
4. **Coordinate stability.** Restore/resize never moves token boxes relative to
   the page; `fit_max_side` runs once at entry and is recorded in metadata.
5. **Opt-in for anything unproven.** The verifier, the mixed router and the
   photo tab stay behind explicit switches until their gates pass.

## Extension points

- **New OCR backend:** implement the `OcrBackend` protocol in `document_ocr.py`
  (`tokens`, `text`, `line_orientation_votes`) and register it; the bake-off
  harness (`evals/harness/eval_models.py`) scores it on the frozen sets.
- **New language:** add the fixture strings and RapidOCR/Tesseract language
  codes, then evaluate on a frozen set before claiming support.
- **New recognizer:** `deva_crnn/` is the template — export data, train, then
  run `scripts/eval_deva_crnn_gate.py`; adoption requires the pre-registered
  gate to pass.
