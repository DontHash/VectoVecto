# Value roadmap — trust → correction loop → data

Planning date: 2026-10-01. Grounded in the frozen-set evidence
([EVALUATION.md](EVALUATION.md), [PLAN.md](PLAN.md) Appendixes N–AF) and the
discoverability plan ([SEO_AEO.md](SEO_AEO.md)). Every item ships only through
its pre-registered gate — or it does not ship.

## 0. Thesis

The highest-leverage gap in the system is not another model. It is the
**missing feedback loop**:

- the review queue is the product's differentiator, but it is **read-only** —
  a user sees "check token ७" and cannot act on it in-product, and their
  corrections never feed the pipeline;
- model-side improvements keep failing for lack of *target-domain* ground
  truth (Appendix AF: 219 mixed crops did not teach court-register runs);
- the failure evidence is consistent: the delta is **structure and formats**,
  not glyph recognition (bagCER is tied across every engine; 43% of modern
  digit conflicts are merged multi-number boxes).

Three compounding bets plus one continuous one:

1. **Immediate trust** — number-run splitting + domain-format validators. No
   new data needed; attacks the measured failure modes.
2. **The correction loop** — make every flag actionable (verify → correct →
   re-export), with an explicit, privacy-first correction export. This is
   simultaneously the killer feature and the data supply chain.
3. **Robustness inputs** — a real phone-photo frozen set (the biggest
   user-facing gap currently has only a synthetic proxy) and targeted
   restores behind the honesty gates.
4. **Evidence & distribution** (continuous) — publish every adopted gain; the
   benchmark is the citable asset that feeds [SEO_AEO.md](SEO_AEO.md).

## 1. Evidence base

| Area | Measured state | Where value sits |
|---|---|---|
| Clean modern print | cornell **0.0541**, gov PDFs 0.135 | solved; stop investing |
| Letterpress default | **0.373** (`auto` reader, Appendix AE) | held; monitor only |
| Modern PDF digit trust | queue digit R@10 **0.38**; digit-exact pages 3/41 | **primary target** |
| Failure shape | 43% of conflicts ≥8-digit merged runs; near-misses (0.66 overlap) | structure, not glyphs |
| Verifier (second reader) | CRNN tracks falsified (0.0 precision, Appendix AF); bodhan VLM 0.81 precision but +8.3 s/page (opt-in) | rules & splitting first |
| Hard table pages | pipeline 0.255 vs raw 0.530; VLMs 0.17 at 21–23× cost | order/structure work |
| Phone photos | proxy only: 0.370 / 0.757 | needs real ground truth |
| Product loop | queue read-only; corrections discarded | **unbuilt asset** |

## 2. Track 1 — Immediate trust (weeks 1–2, no new data)

### 1a. Number-run splitting (the direct structural fix)

**Problem:** 43% of modern digit conflicts are boxes holding 2–3 glued
numbers (`२०८१०४२७२०८१०४३२` — two dates, 16 digits). No reader can fix that;
the input must be split.

**Do:** a segmenter that splits token boxes at wide internal gaps, ruling
lines and column boundaries; each segment is re-read by the engine and enters
reading order and the queue as its own token.

**Gate:** hard-10 page CER ≤ 0.22 (−≥12% relative); v2 digit queue R@10
**+≥3 pp**; byte-identical output elsewhere; invented tokens unchanged.

**Effort:** 3–5 days.

### 1b. Domain-format validators (flags, no model)

**Do:** `invalid_format` flags with reasons — impossible calendar values (the
run showed a day-32 date), malformed case-number patterns, malformed amount
groupings. Flag-only; the alternative never replaces text.

**Gate:** v2 digit queue R@10 **+≥3 pp** with measured flag precision ≥0.5 on
GT; no queue growth on the other frozen sets.

**Effort:** 2–4 days.

### 1c. Queue ranking with structural evidence

Weight validator hits and split uncertainty into `token_risk`.

**Gate:** R@10 improves; R@30 precision does not collapse.

## 3. Track 2 — The correction loop (weeks 2–5, the flywheel)

**Why:** today the queue informs and stops there. Making it actionable is the
product; the same corrections, exported explicitly, are exactly the
target-domain data that model attempts kept lacking.

1. **Review mode in the web studio** — keyboard-first stepping through flagged
   tokens: crop + page context, engine reading, alternative readings, free
   correction.
2. **Apply corrections** — rebuild searchable PDF, transcript, Markdown and
   `ocr.json` with per-token provenance (`corrected_by: human`); CLI parity:
   `--apply-corrections corrections.json`.
3. **Correction export (explicit, local, privacy-first)** — a zip the user
   creates themselves: corrected text + crops of only the tokens they mark
   shareable; no page images; format compatible with the N5 training
   pipeline. No telemetry, nothing leaves the machine without this action.
4. **Metrics:** end-to-end demo; 10 beta runs; ≥20% of queued tokens acted on.

**Why this is the highest-value item:** it *is* the differentiator (verify →
fix → export is what "honest OCR" means in practice), and it converts normal
use into consented ground truth at the scale Appendix AF showed is needed.

## 4. Track 3 — Robustness inputs (weeks 4–10)

### 3a. Real phone-photo frozen set

Consent + anonymization protocol; 50+ real pages; becomes
`nepali_photo_real_v1`. Replaces the synthetic proxy numbers (0.370/0.757)
with real ones and gives every future restore experiment a target.

### 3b. Targeted restore experiments

Dewarp, shadow removal, denoise — each behind the W2.1-style gate that killed
sauvola (invented tokens ≈0, no CER regression). Adopt-with-numbers only.

### 3c. Synthetic in-domain generation (conditional)

Court-register-style number tables (dates, case numbers, amounts) with exact
GT, tens of thousands of samples — only if Track 1 plateaus and a model path
reopens; then the Appendix-AC verifier gate again.

## 5. Track 4 — Evidence & distribution (continuous)

- Re-run and republish `BENCHMARK.md` + README numbers after every adopted
  change; every improvement becomes citable content.
- Launch sequence per [SEO_AEO.md](SEO_AEO.md) §6 once Track 1/2 land
  (Show HN, Product Hunt, community posts with the new numbers).
- Desktop installer (packaging spec exists) → winget/choco listing for the
  offline archive/court segments.

## 6. Sequencing

| Weeks | Track | Dependency |
|---|---|---|
| 1–2 | 1a splitting, 1b validators, 1c ranking | none |
| 2–5 | 2 correction loop (web + CLI + export) | none (parallel) |
| 4–6 | 3a photo-set protocol; first 3b experiments | community access |
| 5–8 | Republish benchmark + launch once 1a/1b land | tracks 1–2 |
| 8–10 | 3c or next structural item, chosen by the gates | track 1 results |

## 7. Success metrics (90 days)

| Metric | Baseline | Target |
|---|---|---|
| Modern-PDF digit queue R@10 | 0.38 | **≥0.55** |
| Hard-10 page CER | 0.255 | ≤0.22 |
| Letterpress default | 0.373 | held, no regression |
| Correction loop | none | demo + 10 beta runs |
| Real-photo set | proxy only | 50+ consented pages |
| Distribution | 0 mentions | per [SEO_AEO.md](SEO_AEO.md) §7 |

## 8. Explicit non-goals

- No more small mixed-crop CRNN fine-tunes (falsified, Appendix AF).
- No VLM default (cost + invented tokens); bodhan stays opt-in.
- No data collection without explicit per-item consent; no telemetry — the
  local-first promise is the brand.
- No metric moves: gates and frozen sets are not renegotiated after a run.

## 9. Guardrails

Unchanged: pre-registered adopt-if gates, frozen sets never used for tuning,
honesty invariants (invented ≈0, alternative readings never silently replace),
≤ +1 s/page budgets for recognition changes, privacy by default.
