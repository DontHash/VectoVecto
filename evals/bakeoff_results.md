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

Notes:
- Line mode = recognition-only on identical crops cut from the frozen pages
  (4px pad, manifest order, first 150 with digits).
- `invented` uses the peer rule: token absent from GT *and* from the RapidOCR
  page text. For the RapidOCR row it is 0 by construction.
- Negative results stay in this table; nothing was tuned on these sets.
