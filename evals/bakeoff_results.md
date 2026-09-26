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
| bodhan indic-ocr | Indic Open Model License 1.0 | lines 150 | 1.210 raw / **0.361 md-stripped** | - | **0.653** | bag invented 0.395 | 0.44 | **PASSES the verifier role** (license self-host OK): digit-exact 0.653 (baseline 0.533); conflict-flag vs baseline digit errors recall **0.714** precision **0.806** (gate 0.4/0.35); 2.15 GB VRAM; output is markdown/LaTeX-wrapped, so its own CER is meaningless without stripping |
| bodhan indic-ocr | (same) | pages 3 (probe) | 0.330-0.541 md-stripped | - | digBAG 1.0-58.0 | - | **329-434 s/page** | **fails the primary role** (gate 8 s/page); quality comparable to baseline, one page catastrophically wrong on digits |

**M4 verdict (bodhan):** no shippable primary recognizer (330-430 s/page on
the HF quickstart path; the vLLM path that would fix this needs a GPU they
don't document for 4 GB cards), but the **first candidate to pass the
verifier gate**: when its digit reading disagrees with RapidOCR's, the
baseline token is wrong 81% of the time and 71% of baseline digit errors are
caught. Integration (opt-in `--digit-verifier`, flag-only, text never
changed) follows in the F4 commit. License obligations for shipping:
attribution/notice, self-hosting allowed, no third-party hosted access,
>500M MAU / >$250M revenue gate.

## Phase 1 — Qwen3-VL: modern documents first (2026-09-26)

Sets: `deva_real_lines` (first 150 of 745 modern line crops, Gemini-labelled)
and the first 10 pages of `nepali_pdf_v2` (born-digital government PDFs,
Gemini-anchor GT). Those 10 pages are the table-heavy hardest slice — on the
full 41-page set RapidOCR scores CER 0.135. Frozen data; nothing tuned here.

| model | license | mode | CER | bagCER | digit-exact / digBAG | invented (vs GT+rapidocr) | s/unit | verdict |
|---|---|---|---|---|---|---|---|---|
| rapidocr (baseline) | Apache-2.0 | lines 150 | **0.0082** | - | **1.000** | 0 | **0.08** | reference |
| rapidocr (baseline) | Apache-2.0 | pages 10 | 0.5295 | 0.3799 | 0.1351 (digBAG) | 0 | 1.51 | reference |
| qwen3vl-4b fp16 (V100) | Apache-2.0 | lines 150 | 0.146 | - | 0.077 | - | 3.46 | **loses** (exact 0.713) |
| qwen3vl-8b-4bit-rt NF4 (T4) | Apache-2.0 | lines 150 | 1.959 mean / **median 0.000** / catastrophic 2% | - | 0.077 | - | 7.20 | **loses** — 72.7% exact vs 92.7%; digits render as **Bengali numerals** (२०७५ → ২০১৫); one runaway repetition (CER 272) drives the mean |
| qwen3vl-8b-4bit-rt NF4 (T4) | Apache-2.0 | pages 10 | **0.176** [0.129-0.235] / median 0.141 / catastrophic 0% | 0.369 | **0.107** (digBAG) | **224 tokens / 30 digit** (rate 0.47) | **152.7** | **quality win, gate fail** — page CER 3× better than baseline (reading order on tables) but ~100× slower and hallucinates |

**Verdict: no VLM mode ships; the classical engine stays the default.**
Neither pre-registered role passes: the primary gate wants ≤8 s/page and
invented ≈0 (measured 152.7 s/page on a T4 and 224 invented tokens on 10
pages), the verifier gate wants digit-exact ≥0.65 (measured 0.077). What is
real: on the hard table pages the VLM's reading order cuts CER 0.53 → 0.18
while bagCER is a tie (0.369 vs 0.380) — the classical weakness there is
structure/order, not glyph reading. The only defensible VLM use is a hybrid
(classical/CRNN lines + digits, VLM as an opt-in page reader for table-heavy
pages, digit verifier as guard) — Phase 5 material, and only on GPUs the
product does not currently require. The shippable path remains bespoke
training (W1 line reader + Phases 2–4).

Reproduce (Kaggle T4, runtime NF4 of the Apache-2.0 checkpoint):
`kaggle/vlm_eval/` kernel + `bhishmbhandari/vectovecto-vlm-eval-data`
dataset; GCP fp16 recipe in `scripts/gcp_vlm_8b_startup.sh`. Infra lessons:
DLVM torchaudio ABI needs the shim; `jinja2>=3.1`; `rapidocr` needs
`onnxruntime`; the in-dir manifest stored Windows separators (fixed —
`doc_data.load_dataset` now normalizes, POSIX-safe); 8B fp16 does not fit a
16 GB V100 (CPU offload ⇒ ~1 h+/page with repetition loops — abandoned);
NF4 needs compute capability ≥7.5 (T4 yes, V100 no).
