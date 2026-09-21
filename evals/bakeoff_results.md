# Devanagari OCR bake-off — results

Frozen sets only (hash-verified): `evals/manifests/heidata_printed_v1.json`
(69 letterpress pages, human ALTO GT; the first 150 digit-bearing lines are
used for line scoring) and `nepali_pdf_v1.json` for page checks.
Baseline to beat: **RapidOCR PP-OCRv5 devanagari mobile**.

Adoption gates (pre-registered):
- verifier role: digit-seq exact >= 0.65 on heiDATA digit lines; conflict-flag
  recall@10 >= 0.4 @ precision >= 0.35; invented tokens vs GT+rapidocr ~ 0;
  permissive license; offline.
- primary role: heiDATA page CER <= 0.30, PDF CER <= 0.25, <= 8 s/page on the
  RTX 2050, invented ~ 0.

| model | license | mode | CER | bagCER | digit-exact / digBAG | invented (vs GT+rapidocr) | s/unit | verdict |
|---|---|---|---|---|---|---|---|---|
| rapidocr (baseline) | Apache-2.0 | lines 150 | 0.316 [0.273-0.360] | - | **0.533** | 0 | 0.10 | reference |
| rapidocr (baseline) | Apache-2.0 | pages 69 | 0.4335 [0.388-0.483] | 0.5529 | 8.98 (digBAG) | 0 | 0.84 | reference |
| tesseract-nep | Apache-2.0 | lines 150 | 0.741 [0.656-0.832] | - | 0.207 | - | 0.16 | **loses** (psm 6/7/13 all >= 0.74 CER on 40-line probe) |
| tesseract-nep | Apache-2.0 | pages 69 | **0.353 [0.277-0.446]** | 0.531 | 3.38 (digBAG) | **5,080 tokens / 532 digit** | 2.60 | page CER beats baseline but **fails the invented gate hard** (classic tesseract garbage; also 3x slower) |
| trocr (MIT) | MIT | lines 150 | **1.043 [0.968-1.139]** | - | **0.020** | - | 0.17 | **loses decisively** — trained on handwritten Nepali words; on printed letterpress lines it hallucinates plausible but unrelated Devanagari |
| glmocr base | Apache-2.0/MIT | lines 4 (probe) | ~1.0 | - | 1/4 exact | - | 1.8-3.1 | line crops are out of distribution (page-level model) |
| glmocr base | Apache-2.0/MIT | pages 3 (probe) | 0.581 / 0.954 / 0.889 | - | - | repetition loops (e.g. 'सरवसरवसरव...') | **196-229** | **loses on both axes**: worse CER than baseline on page 1, degenerate repetition, ~200 s/page on the RTX 2050 (2.26 GB VRAM peak - it *fits*, it is just slow and weak on Devanagari) |

Notes:
- Line mode = recognition-only on identical crops cut from the frozen pages
  (4px pad, manifest order, first 150 with digits).
- `invented` uses the peer rule: token absent from GT *and* from the RapidOCR
  page text. For the RapidOCR row it is 0 by construction.
- Negative results stay in this table; nothing was tuned on these sets.
- GLM-OCR generation tuning (repetition_penalty=1.3, no_repeat_ngram=8,
  max 256 tokens) made page 1 *worse* (CER 0.581 -> 0.700) and mixed Latin
  garbage into Devanagari output; still ~198 s/page. The degeneracy is the
  model's behaviour on this domain, not a sampling flag.
- `himalaya-ai/glm-ocr-devanagari-finetuned` was NOT run: it ships no root
  weights (6 x 3.57 GB checkpoints only, license none) — internal-only and
  pointless while the shippable base fails every gate. Recorded as skipped.
- bodhan-ai/indic-ocr: blocked on `hf auth login` (gated repo; token belongs
  to the user).
