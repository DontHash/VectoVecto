# Devanagari document OCR — head-to-head benchmark

VeriScript's claim is that glyph recognition is a commodity and the delta is
**order, digits and knowing when to flag**. This is the measurement behind
that claim: the shipped pipeline against the raw engine, a full PaddleOCR
pipeline, and the VLM-class candidates (Surya OCR 2, Qwen3-VL-8B), all scored
by the same metric code on frozen, hash-verified sets.

Published 2026-09-30. Raw results and full lab notes:
[`evals/bakeoff_results.md`](../evals/bakeoff_results.md). This page is the
citable summary; the lab notes keep every negative result.

## Sets

| Set | What it is | Ground truth | Redistributed? |
|---|---|---|---|
| `nepali_pdf_v2` | 41 born-digital Nepali government PDFs; the "hard-10" is the table-heavy first 10 pages | Gemini-2.5-Pro anchor protocol | no — internal only |
| `heidata_printed_v1` | 69 letterpress pages (heiDATA, CC BY 4.0, doi:10.11588/data/EGOKEI) | human ALTO transcription | no — pages stay with heiDATA |
| `cornell_real_v1` | 16 clean modern textbook scans | human transcription | no — pages stay with Cornell eCommons |

Every set is a hash-frozen manifest (`evals/manifests/`); `eval_freeze.py
--check` verifies nothing drifted. Nothing below was tuned on these sets, and
negative results stay in the tables.

## Head-to-head — the hard-10 table-heavy slice

The hardest 10 pages of `nepali_pdf_v2` (court registers and dense tables).
CIs are 2000-resample bootstrap over pages. "invented" counts tokens in
neither the ground truth nor the RapidOCR pass (peer rule).

| Arm | Hardware | CER mean [95% CI] | Median CER | bagCER | digBAG | invented | s/page | Licence |
|---|---|---|---|---|---|---|---|---|
| Surya OCR 2 | RTX 2050 4 GB, llama.cpp/Vulkan | 0.166 [0.100–0.258] | **0.107** | 0.360 | 0.105 | 82 | 172.0 | code Apache-2.0; **weights OpenRAIL-M** (free only below $5M) |
| Qwen3-VL-8B NF4 | T4 | 0.176 [0.129–0.235] | 0.141 | 0.369 | 0.107 | 224 | 152.7 | Apache-2.0 |
| **VeriScript pipeline (default)** | RTX 2050, DirectML | **0.255 [0.202–0.329]** | 0.227 | 0.376 | 0.129 | 35 | **7.3** | Apache-2.0 |
| raw RapidOCR engine (PP-OCRv5 deva mobile) | RTX 2050, DirectML | 0.530 [0.463–0.585] | 0.543 | 0.380 | 0.135 | 0 | 3.8 | Apache-2.0 |
| PaddleOCR 3.x full pipeline | CPU | 0.662 [0.585–0.713] | 0.687 | 0.400 | 0.166 | 186 | 21.8 | Apache-2.0 |

**Reading.** `bagCER` — the character distance after ignoring reading order —
is **tied across all five arms (0.360–0.400)**: glyph recognition is
effectively identical, and the whole page-CER spread is ordering, table
structure and digit handling. The shipped pipeline halves the raw engine's
page CER (0.530 → 0.255) at ~2× the engine time, staying CPU-first, offline
and Apache-2.0. Surya 2 reads best by median (0.107) with CIs that overlap
ours, bought with ~23× our page time on the same consumer GPU and a weights
licence that gates commercial use. PaddleOCR's own full pipeline loses to its
own engine on this slice (0.662 vs 0.530) — its ordering on dense tables is
worse and its Devanagari registry has a mobile recognizer only. Both GPU arms
invent text the source never contained (82 and 224 tokens over the 10 pages).

## Letterpress, 69 pages (default settings)

| Arm | CER [95% CI] | bagCER | digit-flag coverage | ECE | s/page |
|---|---|---|---|---|---|
| raw RapidOCR | 0.4335 [0.387–0.481] | 0.553 | 0.589 | 0.818 | 2.0 |
| **VeriScript pipeline** | 0.4309 [0.385–0.480] | 0.548 | **0.616** | **0.810** | 4.8 |
| pipeline + CRNN line reader (opt-in, letterpress only) | **0.253** | — | — | — | — |

Default is parity with the raw engine; the pipeline's gain here is
digit-flag coverage and calibration. The opt-in in-repo CRNN reader
(`--deva-lines`) cuts letterpress CER to 0.253, but it is letterpress-specific
— on clean modern scans it measurably *hurts* (0.054 → 0.156), so it stays
off by default.

## Clean modern scans

`cornell_real_v1` (16 pages, human GT): pipeline CER **0.0541 [0.036–0.076]**
— prose 0.007–0.049, table-of-contents pages 0.073–0.153. Clean print is a
solved case; degradation, tables and digits are the remaining gap.

## The trust half — the review queue

Accuracy is one axis; the other is what happens to the digits that are wrong.

| Signal | Result |
|---|---|
| 2× digit re-pass (default on for Devanagari) | queue digit recall@10 **0.73 → 0.88** letterpress, **0.15 → 0.38** PDFs |
| Isotonic `cal_conf` calibration | ECE **0.818 → 0.297** letterpress, 0.821 → 0.411 PDFs (dev 0.640 → 0.105) |
| Optional `unknown_word` lexicon flag | frozen-PDF token R@10 0.148 → **0.175** (P@10 0.751 → 0.755); neutral on letterpress |

Every flagged token keeps its alternative reading beside it, and the text is
never silently changed — the conflict is the product. None of the compared
arms exposes alternative readings at all.

## Limitations (honest)

- The PDF set's ground truth is a Gemini-2.5-Pro anchor protocol, not human
  transcription; it is consistent across arms but not infallible.
- The frozen sets are internal-only (licences above), so reproducing the PDF
  numbers requires the same local data. **No cloud-API arm yet**: sending the
  internal PDFs to a vendor needs either a public Devanagari set or explicit
  permission — both are open items.
- Box-coverage figures for the pipeline on pages larger than 2500 px are under
  investigation (coordinate-space artifact suspected); not quoted here.
- Phone photos are measured on a synthetic proxy (CER 0.370 medium / 0.757
  heavy), not a real photographed field set — that field set does not exist
  yet.
- Devanagari digits on real scans remain the weakest link; the N2 verifier
  experiment failed its precision gate and is not shipped
  ([`docs/PLAN.md`](PLAN.md) Appendix AC).

## Reproduce

```bash
# ours + raw (frozen-verified, bootstrap CIs)
python evals/harness/eval_document.py --data-dir data/doc_eval/nepali_pdf_v2 \
  --pages 10 --frozen evals/manifests/nepali_pdf_v2.json --lang ne \
  --ocr rapidocr --methods raw,pipeline --bootstrap 2000 \
  --json evals/headtohead_nepali_pdf_hard10.json

# Surya 2, PaddleOCR and the letterpress runs: commands and pins are in
# evals/bakeoff_results.md ("Reproduction") — raw JSON next to this repo's
# evals/: headtohead_nepali_pdf_hard10.json, headtohead_heidata69.json,
# surya_nepali_pdf_hard10.json, paddleocr_nepali_pdf_hard10.json
```

Frozen manifests and metric code: `evals/manifests/`,
`veriscript/core/doc_metrics.py`. Set licences: [`LICENSES.md`](LICENSES.md).
