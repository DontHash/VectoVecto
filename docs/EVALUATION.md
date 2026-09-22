# Evaluation

Every number in the README comes from a **content-hash frozen** evaluation set
with bootstrap 95% confidence intervals. This document is the map: what the
sets are, how to verify them, how to reproduce the numbers, and what the known
limitations are.

## Frozen sets

| Manifest | Set | Source & license | GT quality |
|---|---|---|---|
| `evals/manifests/heidata_printed_v1.json` | 69 letterpress pages, 7 books, 1451 lines | heiDATA doi:10.11588/data/EGOKEI, CC BY 4.0 | human-corrected Transkribus ALTO (text + line boxes) |
| `evals/manifests/heidata_dev_v1.json` | 61 pages, 4 books, 1322 lines | same collection, held out for tuning | human-corrected ALTO. **Consumed as training data by W1 attempt 2** — no longer a tuning set |
| `evals/manifests/nepali_pdf_v2.json` | 41 born-digital pages, 6 documents | supremecourt.gov.np / lawcommission.gov.np (internal-only) | Gemini 2.5 Pro anchor, engine-corroborated; text layers rejected as corrupt |
| `evals/manifests/nepali_pdf_v1.json` | 41 pages, first version | same | corrupt text layer — historical only, superseded by v2 |
| `evals/manifests/nepali_lines_v1.json` | 500 real line crops | HF `himalaya-ai/nepali-deva-ocr-eval` (unknown provenance) | machine-generated, partly misaligned → behavior-only numbers |
| `evals/manifests/nepali_photo_proxy_v1.json` | 41 pages, camera artifacts | derived from `nepali_pdf_v2` | anchor GT, labeled **proxy** for real photos |

Discipline: frozen sets are never used to tune thresholds. A dataset change is
a new freeze version, not an edit.

## Verify the sets

```bash
python evals/harness/eval_freeze.py --check evals/manifests/heidata_printed_v1.json
```

The check re-hashes every recorded file; drift aborts the evaluation harness.

## Reproduce the headline numbers

```bash
# Letterpress pages (human ALTO GT) — page CER + digit metrics
python evals/harness/eval_document.py --data-dir data/doc_eval/heidata_printed \
    --frozen evals/manifests/heidata_printed_v1.json --lang ne --methods raw \
    --bootstrap 2000

# Modern government PDFs against the Gemini anchor
python evals/harness/eval_document.py --data-dir data/doc_eval/nepali_pdf_v2 \
    --frozen evals/manifests/nepali_pdf_v2.json --lang ne --methods raw \
    --bootstrap 2000

# Real line crops (behavior only)
python evals/harness/eval_lines.py --data-dir data/doc_eval/nepali_lines \
    --frozen evals/manifests/nepali_lines_v1.json --lang ne --bootstrap 2000

# Review-queue metrics (digit + all-token recall@10)
python evals/harness/eval_flags.py --data-dir data/doc_eval/heidata_printed \
    --lang ne --frozen evals/manifests/heidata_printed_v1.json

# Error taxonomy (where recognition fails: segmentation vs recognition)
python evals/harness/eval_error_taxonomy.py --data-dir data/doc_eval/heidata_printed \
    --frozen evals/manifests/heidata_printed_v1.json --lang ne

# Model bake-off (scores any registered backend on the frozen sets)
python evals/harness/eval_models.py --list
```

Raw data lives under `data/doc_eval/` (git-ignored). Rebuild commands are
recorded in each manifest and in [PLAN.md](PLAN.md) Appendix K.

## Ground-truth integrity

No set is trusted without an audit:

- `doc_metrics.devanagari_validity()` measures invalid combining sequences
  (reordered matras, dangling viramas, orphan marks).
- `scripts/harvest_nepali_pdfs.py` rejects PDFs whose text layer scores >2%
  invalid tokens (modern Nepali PDFs routinely have corrupt ToUnicode maps).
- Where no clean reference exists, the anchor is model-produced and labeled:
  `evals/harness/anchor_gemini.py` transcribes pages with Gemini 2.5 Pro
  (Vertex AI, ADC), the local engines corroborate, and the disagreement list
  is the human review artifact (`out/anchor_gemini/audit.html`).

## Adopt-if gates (pre-registered)

| Change | Gate | Result |
|---|---|---|
| 2× digit re-pass default (W0.3) | digit R@10 ≥ +3pp on frozen heiDATA | PASS — 0.73 → 0.88 letterpress, 0.15 → 0.38 PDFs; default ON for Devanagari |
| Digit verifier (bodhan) auto-enable | queue R@10 +≥3pp and ≤ +1 s/page | FAIL on cost (+8.3 s/page) → stays opt-in |
| Mixed-page router (P5) | text CER +0%, non-text PSNR pass, no invented tokens | 1 invented token on photo texture → stays opt-in |
| Letterpress preprocessing (W2.1) | CER −≥5% relative | FAIL — sauvola invents 1754 tokens; raw stays |
| Devanagari line recognizer (W1) | digit-exact ≥0.75 on frozen heiDATA lines | FAIL so far — best **0.719** (h=48 Kaggle run + ensemble), single model 0.710; RapidOCR remains the engine but the recognizer is now 2.6× RapidOCR on digit-exact (0.719 vs 0.278) and better on line CER. Trajectory: [PLAN.md](PLAN.md) Appendix R2–R4 |

## Known limitations (with evidence)

- **Digit reading on real scans** is the weakest link: Devanagari digits are
  treated as unread (mobile rec model) and surfaced through the review queue.
- **Token flags on real Devanagari** carry little information (coverage 0.002
  on PDFs, 0.016 on letterpress; ECE 0.82) — the queue, not the colors, is the
  honest signal.
- **Line-crop GT** (`nepali_lines`) is machine-generated and partly misaligned;
  numbers from it are behavior-only.
- **Phone photos** are measured on a synthetic proxy, not a real field set:
  CER 0.370 medium / 0.757 heavy.
- **>2 columns** are unsupported; the router and layout logic assume one or two.

Full history, per-appendices, in [PLAN.md](PLAN.md); the bake-off table is in
[../evals/bakeoff_results.md](../evals/bakeoff_results.md).
