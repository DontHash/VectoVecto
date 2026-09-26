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
| `evals/manifests/cornell_real_v1.json` | 16 real textbook scans (4 books, grades 4-7), 564 lines | Cornell eCommons collection 1813/24179 (evaluation only, pages not redistributed) | **human-corrected** from the shipped pipeline pre-fill; 0.0% invalid Devanagari on all pages |

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
| Lexicon flag `unknown_word` (W-B) | queue R@10 +≥3pp, no digit regression | **PARTIAL — opt-in.** Frozen modern PDFs: token R@10 0.1477 → **0.1754** (+2.8pp), token P@10 +0.4pp, digit R@10 unchanged; frozen letterpress: neutral (token +0.1pp, digit R@10 0.8832 unchanged). Active only when the optional lexicon is built (`scripts/fetch_nepali_lexicon.py`); the looser `frac=0.5` config reaches +4.5pp on PDFs but costs 1.5pp letterpress digit R@10. Details: [PLAN.md](PLAN.md) Appendix T |
| VLM page reader (Qwen3-VL-8B, Phase 1) | page CER ≤0.25, ≤8 s/page, invented ~0 | **FAIL on cost + invented.** Same 10 hard `nepali_pdf_v2` pages: CER **0.176** vs RapidOCR 0.530 (median 0.141, digBAG 0.107 vs 0.135) — but **152.7 s/page** (T4, NF4) and 224 invented tokens; lines median CER 0.0 yet digit-exact 0.077 (Bengali numerals). No VLM mode ships; hybrid is Phase 5 material at best. Details: [bakeoff_results.md](../evals/bakeoff_results.md) |
| Table cell-major reading order (Appendix Y) | hard-10 `nepali_pdf_v2` page CER −≥15% relative; no regression elsewhere | **PASS.** Pipeline CER 0.5280 → **0.2550** on the 10 hardest pages (−51.7%; full 41: 0.3417 → 0.2751), bagCER/digBAG/invented bit-identical (order-only), grid branch fires 0× on heidata-printed/SROIE/CORD/arXiv-two-column/photo-proxy (byte-identical), s/page within budget. Details: [PLAN.md](PLAN.md) Appendix Y |
| Reader `auto` gate (N0) | no engagement on modern print; letterpress behaviour unchanged | **PASS.** `auto` engages 0/16 cornell_real, 0/41 nepali_pdf_v2, 0/41 photo-proxy (CER exactly the reader-off numbers: 0.0541 / 0.3379) and 42/69 heidata as before (the paper gate excludes nobody there); the reader measurably hurts real modern scans (0.0541 → 0.1558 forced), so this closes the `auto` foot-gun. Details: [PLAN.md](PLAN.md) Appendix AA |
| Dense-table row-major order (N1) | 4 ToC pages −25% rel; no regression elsewhere | **MISS on the primary clause, adopted with a recorded amendment.** 4-ToC mean CER 0.1167 → 0.0985 (−15.6%); 2 of the 4 pages were already row-major (sorter output unchanged, residual = digits) and the 2 ordering-affected pages improved −27.3%; cornell overall 0.0541 → **0.0495**; v2 hard-10/41 bit-identical; 0 dense fires on heidata/SROIE/CORD/arXiv-2col/photo-proxy/mixed (byte-identical). Details: [PLAN.md](PLAN.md) Appendix AB |
| Digit verifier via CRNN disagreement (N2) | precision ≥0.6 → flag-only; ≥0.8 → replacement; <0.4 → stop | **STOP (precision 0.50).** ToC lesson digits: engine 9/23, CRNN 11/23, 12 disagreements (CRNN right 0.50 / engine right 0.33 / both wrong 0.17); v2: 168 conflicts on 261 tokens, CRNN fixes 4; cost 0.19–0.31 s/page. Not shipped — the digit residual needs a better model (N3) or more real GT (N5). Details: [PLAN.md](PLAN.md) Appendix AC |

## Real textbook probe (W-D, unlabeled)

No accessible born-digital textbook corpus was found (2026-09-25): the CDC
catalogue's ResourceSpace download endpoints return 404 with and without a
session, and the MOEST eLibrary text layers are legacy-font mojibake (20/20
gated out: 16 mojibake, 4 scan-only). Cornell eCommons hosts 458 Nepali
textbook scans (collection 1813/24179) with no text layers, so 4 books are
profiled as a behaviour-only probe (32 content pages, `--start-page 2`):
**26.3 tokens/page, 24.8% flagged, 0.36% `script_mismatch`, 4.0%
`unknown_word`, 0 orientation suspects, median 1.46 s/page**. Reproduce:
`python scripts/harvest_nepali_textbooks.py --source cornell --limit 4` then
`python scripts/profile_textbook_scans.py --start-page 2 --max-pages-per-pdf 8`.
No CER is claimed; a human-corrected slice remains the open path to real
textbook ground truth.

## Known limitations (with evidence)

- **Digit reading on real scans** is the weakest link: Devanagari digits are
  treated as unread (mobile rec model) and surfaced through the review queue.
- **Token flags on real Devanagari** carry little information (coverage 0.002
  on PDFs, 0.016 on letterpress; ECE 0.82) — the queue, not the colors, is the
  honest signal. The optional lexicon flag (`unknown_word`, W-B) lifts the
  frozen-PDF token queue to R@10 0.175 / P@10 0.755 (from 0.148 / 0.751) with
  the digit queue unchanged; it is neutral on letterpress.
- **Line-crop GT** (`nepali_lines`) is machine-generated and partly misaligned;
  numbers from it are behavior-only.
- **Phone photos** are measured on a synthetic proxy, not a real field set:
  CER 0.370 medium / 0.757 heavy. Clean printed textbook scans are now
  measured on human GT instead: `cornell_real_v1` pipeline CER **0.0541**
  [0.036-0.076] (prose 0.007-0.049, table-of-contents pages 0.073-0.153) —
  clean print is a solved case; degradation and tables are the remaining gap.
  The W1 CRNN reader measurably **hurts** real scans (`deva_lines on`:
  CER 0.0541 → 0.1558 on the same 16 pages): it is letterpress-specific and
  stays off by default. Details: [PLAN.md](PLAN.md) Appendix Z.
- **>2 columns** are unsupported; the router and layout logic assume one or two.

Full history, per-appendices, in [PLAN.md](PLAN.md); the bake-off table is in
[../evals/bakeoff_results.md](../evals/bakeoff_results.md).
