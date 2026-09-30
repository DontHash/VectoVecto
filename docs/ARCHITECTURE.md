# Architecture

VeriScript is a local, single-process pipeline. One page in, four artifacts
out; no services, no queue, no network.

```
                       ┌──────────────────────────────────────────────┐
  photo / scan / PDF ─▶│ document/pipeline.run_document_pipeline      │
                       │                                              │
                       │  1 document/orientation  EXIF + 0/90/180/270 │
                       │  2 document/layout       columns, reading    │
                       │                          order               │
                       │  3 document/ocr          RapidOCR/Tesseract  │
                       │                          + digit re-pass      │
                       │  4 document/verifier     optional 2nd model  │
                       │  5 calibration           isotonic conf       │
                       │  6 document/router       text vs non-text    │
                       │                          (opt-in)            │
                       └───────────────┬──────────────────────────────┘
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │ document/export                              │
                       │  searchable PDF · overlay · transcript · JSON│
                       └──────────────────────────────────────────────┘
```

All paths above are inside the `veriscript/` package; the root `cli.py` is a
launcher and the console script is `veriscript`.

## Module map

### Pipeline (document mode)

| Module | Responsibility |
|---|---|
| `veriscript/document/pipeline.py` | orchestrates one page (and multi-page runs); resolves the digit re-pass default per language; keeps coordinate space stable across restore/resize |
| `veriscript/document/ocr.py` | backend registry (`rapidocr`, `tesseract`), token model, risk flags, digit re-pass, verifier hook |
| `veriscript/document/orientation.py` | EXIF plus OCR-evidence rotation (0/90/180/270); refuses a blind 180 flip, flags `orientation?` instead |
| `veriscript/document/layout.py` | column detection and reading order (single-column is identity) |
| `veriscript/document/router.py` | opt-in mixed-page router: restores text regions, leaves logos/photos/signatures untouched |
| `veriscript/document/restore.py` | restore stream (denoise/contrast) with quality gates |
| `veriscript/document/verifier.py` | optional second-model digit re-read (`--digit-verifier bodhan`); disagreements become `cross_model_conflict`, text is never silently changed |
| `veriscript/document/export.py` | searchable PDF (reportlab), numbered review overlay, transcript, OCR JSON; multi-page combined PDF |
| `veriscript/calibration.py` | isotonic confidence mapping (`cal_conf`) fitted on a dev set |
| `veriscript/lexicon.py` | optional Devanagari word list (built by `scripts/fetch_nepali_lexicon.py`) behind the `unknown_word` review flag; absent = flag off |
| `cli.py` → `veriscript/cli.py` | command line: `--mode document|photo`, batch folders, all knobs |
| `webapp/` | the studio: FastAPI backend (`vvweb/`) + SolidJS frontend; web entry point for the document pipeline (photo studio planned) |
| `veriscript/logging_setup.py` | one stderr logger (`vectovecto.*` internal name, `VERISCRIPT_LOG_LEVEL` with legacy `VECTOVECTO_LOG_LEVEL` fallback, default INFO) configured by both entry points |

### Data and metrics

| Module | Responsibility |
|---|---|
| `veriscript/core/data.py` | dataset IO (`load_dataset`, `imread_safe`), PDF→pages, synthetic fixtures, heiDATA/ALTO ingestion, photo-proxy builder |
| `veriscript/core/metrics.py` | CER/WER/bagCER, digit-string metrics, `devanagari_validity`, queue stats, bootstrap CIs |
| `veriscript/core/degradation.py` | deterministic page degradations (mild/medium/heavy) used by proxies and tests |

### Photo/vector stack (parked; CLI photo mode only)

`veriscript/photo/upscaler.py` (content analysis + routing),
`veriscript/photo/sr_engine.py` (ONNX/torch model runner),
`veriscript/photo/srvggnet.py` (SRVGG architecture),
`veriscript/photo/rrdbnet.py` (RRDBNet, x4plus),
`veriscript/photo/hybrid.py` (raster→vector hybrid output). Not part of
the document product path — the weights are non-commercial and no measured
document use case needs it (see [PLAN_WEB_FULL.md](PLAN_WEB_FULL.md),
*Parked for later*).

### Evidence and research (not part of the shipped runtime)

| Path | Responsibility |
|---|---|
| `evals/harness/` | evaluation harnesses: frozen-set verification, metrics, anchor transcription, model bake-off, gates |
| `evals/manifests/` | content-hash frozen evaluation sets (the numbers in the README come from these) |
| `scripts/` | data acquisition (`harvest_*`), training-data export, real-line labeling, recognizer gate, license/release tooling |
| `deva_crnn/` | Devanagari line recognizer (CRNN+CTC): the opt-in reader behind `--deva-lines`; weights are not distributed |
| `kaggle/w1_train/` | private-script kernel that trains the line recognizer (see [TRAINING.md](TRAINING.md)) |

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

- **New OCR backend:** implement the `OcrBackend` protocol in `veriscript/document/ocr.py`
  (`tokens`, `text`, `line_orientation_votes`) and register it; the bake-off
  harness (`evals/harness/eval_models.py`) scores it on the frozen sets.
- **New language:** add the fixture strings and RapidOCR/Tesseract language
  codes, then evaluate on a frozen set before claiming support.
- **New recognizer:** `deva_crnn/` is the template — export data, train, then
  run `scripts/eval_deva_crnn_gate.py`; adoption requires the pre-registered
  gate to pass.
