# Market landscape and positioning

Research date: 2026-09-30. Prices/benchmarks move — verify before quoting.
Sources are listed at the end; vendor claims are marked as such.

## 1. What is ours, and what is borrowed

The product is **not** an OCR engine. The recognizer is a commodity: RapidOCR
(Apache-2.0) wrapping PaddleOCR PP-OCRv5/v6 ONNX models (Apache-2.0); Tesseract
5 is the optional second engine (Apache-2.0). PDF reading/writing is
`pypdfium2` + `reportlab`. None of that is our contribution.

**What this repository builds on top of that commodity:**

| Layer | What it does | Evidence |
|---|---|---|
| Orientation | EXIF + OCR-evidence voting over 0/90/180/270; refuses blind 180 flips and flags instead | 16/16 synthetic; 0/30 false rotations; 90/90 rotated pages decided ([EVALUATION.md](EVALUATION.md)) |
| Layout & reading order | column detection, two-column order, dense-table row-major, court-register cell-major | two-column WER 0.870 → 0.268 |
| Restore + dual-stream audit | classical restore stream vs raw page, digit-by-digit cross-stream comparison; conflicts become flags, text never silently changed | measured queue behaviour, below |
| **Digit honesty** | 2× digit re-pass, risk ranking, isotonic calibration, conflict flags with the alternative reading kept, impossible-sequence / script / lexicon flags, the review queue | queue digit recall@10 **0.88** letterpress / 0.38 PDFs; ECE 0.82 → 0.30 on scans |
| Own recognizer (opt-in) | CRNN+CTC trained in-repo on harvested letterpress lines (`deva_crnn/`), gated by a paper heuristic | letterpress page CER 0.434 → 0.253 with the reader on |
| Exports | searchable PDF with an invisible Unicode layer (bundled Mukta), numbered review overlay, transcript, OCR JSON, multi-page combined, Markdown | frozen-set tested |
| Evaluation infrastructure | hash-frozen manifests, anchor GT protocol, bootstrap CIs, pre-registered gates, negative results kept on file | `evals/` + [PLAN.md](PLAN.md) |
| Product | offline single-process pipeline, CLI + web studio, Docker, license-safe stack, no telemetry | this repo |

One sentence for the pitch: **RapidOCR reads glyphs; we decide what to trust,
in what order, at what orientation — and we prove the behaviour on frozen sets
with published confidence intervals.** The delta over "just run RapidOCR" is
the part that matters for bills and registers: reading order, digits, and
knowing when to say *check this token*.

## 2. Landscape

### 2.1 Cloud OCR APIs (accuracy strong, data leaves the machine)

| Vendor / tier | Price / 1,000 pages | Notes |
|---|---|---|
| AWS Textract — detect text | $1.50 (first 1M), $0.60 above | forms $50, tables $15; free tier 3 months |
| Google Document AI — Enterprise OCR | $1.50 (to 5M), $0.60 above | Cloud Vision text: same rate, 1k units/mo free forever |
| Azure Document Intelligence — Read | $1.50 (to 1M), $0.60 above | Layout $10; prebuilt $10; custom $30–50; 500 pages/mo free |
| Mistral OCR (`mistral-ocr-latest`) | $1.00, ≈$0.50 batch | vendor benchmark: 94.9 overall, hi 97.55 (their eval) |
| LLM vision (Gemini/Claude/GPT-class) | token-priced, very cheap for short pages | strong reading order; hallucination risk unbounded unless gated |

None of the major vendors publish **Nepali/Devanagari page-level CER with
confidence intervals**; benchmarks are internal and self-reported. Our own
frozen-set numbers (CER 0.135 modern PDFs, 0.434 letterpress) are the kind of
evidence missing from this market.

### 2.2 Open-source engines and pipelines

| Project (stars, license) | Role | Devanagari | Honesty/review | Notes |
|---|---|---|---|---|
| PaddleOCR (90k, Apache-2.0) | engine + PP-Structure/VL | yes — PP-OCRv5 devanagari rec 84.96% (their internal set); PP-OCRv6 released 2026-05 | confidence only | the engine we already build on |
| RapidOCR (8k, Apache-2.0) | ONNX Runtime wrapper for Paddle models | as Paddle | confidence only | our recognizer back-end |
| Tesseract (77k, Apache-2.0) | engine | `nep`/`hin` traineddata | confidence only | our own bake-off: page CER **0.353** but **5,080 invented tokens** on 69 letterpress pages and 3× slower — fails the honesty gate |
| OCRmyPDF (35k, MPL-2.0) | searchable PDF via Tesseract | inherits Tesseract | none | closest free analogue to our PDF export half |
| Surya OCR 2 (21k) | 650M VLM-class OCR + layout + tables | yes (Hindi 82.2% on their 91-lang eval) | confidence only | code Apache-2.0, **weights modified OpenRAIL-M (free only <$5M)** — a commercial catch. **We ran it locally** on the frozen hard-10: CER 0.166 (median 0.107) at **172 s/page** on a 4 GB laptop GPU — best text on the table-heavy slice, ~23× our page time |
| Marker (40k, Apache-2.0) | PDF → markdown | weak/unverified | none | Datalab's sibling project |
| Docling (68k, MIT) | document → structured data for LLMs | via OCR back-ends | none | integration hub, not an OCR engine |
| MinerU (81k, license "other") | PDF → markdown/JSON | via OCR models | none | AGPL-class terms — verify upstream |
| olmOCR (20k, Apache-2.0) | linearize PDFs for LLM corpora | English-centric | none | dataset/LLM tooling |
| bodhan-ai/indic-ocr | Indic VLM digit verifier | Devanagari-trained | — | we integrate it as an **opt-in verifier**; it passed our verifier gate (precision 0.81 / recall 0.71 on digit conflicts) but 330–430 s/page fails the primary role |

### 2.3 Commercial desktop / IDP suites

| Product | Price | Notes |
|---|---|---|
| ABBYY FineReader PDF | $99/user/yr Standard; $165 Corporate; Mac $69 | desktop, offline, 190+ languages, human-verification UI; the closest full-feature incumbent |
| ABBYY Vantage / FlexiCapture | custom; benchmark deals $40–100k/yr | cloud IDP, no published Devanagari CER |
| Adobe Acrobat Pro (OCR) | ≈$20/mo | editing suite; OCR is a feature, not the product |
| Online converters (Smallpdf, iLovePDF, …) | freemium | black-box OCR (mostly Tesseract-class); privacy trade-off |

## 3. Aspect comparison

**Accuracy.** On clean Latin print, cloud APIs and modern VLMs are at or above
traditional engines; nobody credibly beats them across scripts. On Devanagari
the published data is thin. Our contribution is not beating them — it is
*measuring* ourselves honestly and rejecting models that look good on vendor
benchmarks. Head-to-head on the frozen hard-10 (2026-09-30, same metric code
— `evals/bakeoff_results.md` Phase 2): bag-of-tokens CER is **tied across all
arms (0.360–0.380)** — glyph recognition is a commodity; the page-CER spread
comes from ordering and structure. Surya 2 leads by median (0.107 vs our
0.227) with overlapping CIs, at ~23× our page time and under OpenRAIL-M
weights; Qwen3-VL-8B was 3× better than the old engine on table pages but 152
s/page with 224 invented tokens; Tesseract invented 5,080 tokens on the
letterpress set and TrOCR hallucinated on printed lines — neither ships.

**Cost.** Cloud read-tier OCR is $1.00–1.50 per 1,000 pages; structured
extraction $10–50; desktop $69–165/user/yr; desktop IDP platforms
$40k+/yr. Self-hosting our document stack: no per-page fee — one CPU process
(~1 s/page clean, 2–4 s letterpress with re-pass). On a $20/mo VPS, even 5%
utilisation ≈ 85k pages/mo ≈ **$0.0002–0.0004/page**, ~5× cheaper than the
cloud read tier before egress/retention costs are counted.

**Performance.** Cloud scales horizontally; our web app is deliberately one
heavy worker (queue + 503 shedding) for a demo/small-office profile. The CLI
is batch. VLM-class competitors need GPUs for comparable speed (ours runs on
CPU; 4 GB DirectML is enough).

**Robustness.** Rotation: measured, decided from evidence, refuses blind
flips. Two-column/tables: measured improvements. Phone photos: honest numbers
(CER 0.370 medium / 0.757 heavy on the synthetic proxy) — recognition, not
honesty, is what breaks there. Handwriting and arbitrary forms: we are weak;
that is cloud/VLM territory.

**Privacy / compliance.** Everything runs on your machine: no data leaves it,
no account, no telemetry — the differentiator for government registers,
health, legal and archive material against every cloud option (ABBYY desktop
is the only widely-known offline peer).

**Licensing / lock-in.** Code MIT; engine Apache-2.0; no AGPL (PyMuPDF
explicitly excluded); weights we ship are either Apache/BSD-class or excluded
from commercial builds by the license gate. Surya-class alternatives carry
non-commercial weight clauses; cloud hosts carry ToS and per-page lock-in.

**The review queue.** No mainstream OCR ships *alternative readings for
conflicting digits*; the nearest analogues are ABBYY's verification UI and
enterprise IDP human-in-the-loop consoles. Combined with isotonic calibration
(ECE 0.82 → 0.30) and a risk-ranked queue with measured recall, this is the
most defensible product feature we have.

## 4. Where we stand

**We win when:** the document is Devanagari (or English) and the digits
matter; the data cannot leave the building; the budget is a machine, not a
per-page fee; the user can afford a 30-second human review of 10 flagged
tokens but not a silently wrong ₹1,20,000 on a bill.

**We lose when:** throughput is measured in millions of pages/day (cloud,
Tesseract farms); the script is outside en/ne/hi; the task is handwriting or
form-field extraction (VLM/IDP suites); the buyer wants a brand, a 99% SLA
and a sales team (ABBYY/Google).

**Threats to watch:** Surya 2 / PaddleOCR-VL-1.5 class models closing the
accuracy gap while staying cheap; cloud read-tier prices already at $1/1k and
falling; VLM inference costs collapsing — the moat is *not* the recognizer.

**Recommendations:**

1. Keep the recognizer swappable behind the honesty layer — that architecture
   is the durable asset, not RapidOCR.
2. Publish a head-to-head Devanagari benchmark (our frozen sets + a cloud API
   + Surya/Tesseract + our queue) once — nobody in this market publishes
   page-level Devanagari CER with alternative readings; this is marketing the
   competitors cannot copy cheaply.
3. Distribution focus: Windows desktop / offline installer (packaging spec
   exists) for archives, courts, municipalities; the web studio for demos.
4. When adopting a stronger back-end (e.g. PaddleOCR-VL, Apache-2.0), run it
   through the same pre-registered gates; the bake-off harness already does
   this.

## Sources

- AWS Textract pricing — aws.amazon.com/textract/pricing
- Google Cloud Document AI pricing — cloud.google.com/products/document-ai/pricing
- Azure Document Intelligence pricing — azure.microsoft.com/pricing/details/document-intelligence (rates corroborated 2026-08 by beri.net and learn.microsoft.com answers)
- Mistral OCR — mistral.ai/news/mistral-ocr (2025-03; benchmark table vendor-run)
- Surya OCR 2 — datalab.to/blog/surya-2 (2026-05; internal 91-language eval)
- PaddleOCR PP-OCRv5 multilingual — paddleocr.ai (devanagari rec 84.96% on their set)
- ABBYY FineReader pricing — pdf.abbyy.com/pricing; PCMag review 2026
- Repository evidence — [EVALUATION.md](EVALUATION.md), [PLAN.md](PLAN.md), `evals/bakeoff_results.md`
- Phase-2 head-to-head (2026-09-30) — `evals/bakeoff_results.md` Phase 2; raw JSON: `evals/headtohead_nepali_pdf_hard10.json`, `evals/headtohead_heidata69.json`, `evals/surya_nepali_pdf_hard10.json`
