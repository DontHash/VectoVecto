# Evaluation sets and results

This folder is the evidence base: frozen manifests, harnesses, and the results
of model bake-offs. Nothing here is used at runtime by the product.

```
evals/
├── manifests/          content-hash frozen sets (verify with eval_freeze.py)
├── harness/            evaluation harnesses and gates
└── bakeoff_results.md  alternative-model scores on the frozen sets
```

## Manifests

| Manifest | What it is | GT |
|---|---|---|
| `heidata_printed_v1.json` | 69 letterpress pages / 7 books / 1451 lines | human-corrected ALTO (CC BY 4.0) |
| `heidata_dev_v1.json` | 61 pages / 4 books — held out for tuning, now consumed as training data by W1 attempt 2 | human-corrected ALTO |
| `nepali_pdf_v2.json` | 41 born-digital government pages | Gemini 2.5 Pro anchor, engine-corroborated |
| `nepali_pdf_v1.json` | first version, corrupt text layer | historical only |
| `nepali_lines_v1.json` | 500 real line crops (unknown provenance) | machine-generated, behavior-only |
| `nepali_photo_proxy_v1.json` | 41 pages with synthetic camera artifacts | anchor GT, labeled proxy |

Verify a set before trusting it:

```bash
python evals/harness/eval_freeze.py --check evals/manifests/heidata_printed_v1.json
```

## Harnesses

| Harness | Purpose |
|---|---|
| `eval_document.py` | page-level CER/bagCER/digit metrics with bootstrap CIs |
| `eval_lines.py` | line-crop evaluation (`--gt-audit` for validity) |
| `eval_flags.py` | review-queue metrics (digit + all-token recall@10) |
| `eval_error_taxonomy.py` | failure classification (segmentation vs recognition vs missing) |
| `eval_anchor.py` + `anchor_gemini.py` | anchor construction: Gemini transcription, engine worksheet, human review |
| `eval_models.py` + `bakeoff_models.py` | score any registered OCR backend on the frozen sets |
| `eval_mixed.py` | mixed-page router gates (text CER, non-text PSNR, cost) |
| `eval_sanity.py` | behavior checks on unlabeled real scans |
| `eval_freeze.py` | freeze/verify manifests (content hashes) |

Usage examples for every harness are in [../docs/EVALUATION.md](../docs/EVALUATION.md).
