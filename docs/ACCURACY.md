# Accuracy roadmap

Research date: 2026-10-01. Grounded in the frozen-set evidence
([EVALUATION.md](EVALUATION.md), [PLAN.md](PLAN.md) appendices N–AC, the
bake-off table in `evals/bakeoff_results.md`). Nothing here is a promise until
it passes its own pre-registered gate — that discipline is what makes the
numbers publishable.

> Program-level sequencing (trust → correction loop → data) is in
> [ROADMAP.md](ROADMAP.md).

## 1. What is solved, and what actually fails

| Class | Measured now | Reading |
|---|---|---|
| Clean modern print | cornell_real CER **0.0541**; gov PDFs 0.135 | Recognition is not the bottleneck here |
| Hard table pages | pipeline **0.255** vs raw 0.530; Surya 0.166 / Qwen 0.176 at **21–23× our time** | bagCER is tied across all arms (0.36–0.40): the gap is **order + structure + digits**, not glyphs |
| Letterpress | **0.434** default; reader `auto` 0.373; forced-on 0.253 | The reader helps; the default is conservative |
| Digits — letterpress | digit CER 8.95 (Latin insertions), queue R@10 **0.88** | Recognition near nil, queue catches most |
| Digits — modern PDFs | digit CER 0.126, whole-page digit-exact **3/41**, queue R@10 **0.38** | Confidently wrong — the moat's weakest link |
| Phone photos | proxy CER 0.370 / 0.757 — **synthetic only** | Biggest user-visible gap; no real ground truth yet |
| Token flags | ECE 0.82 → 0.30 scans; coverage 0.002–0.016 | Weak alone; lexicon adds +2.8 pp on PDFs |
| >2 columns / forms | unsupported | Scoped out, unchanged |

The strategic frame from [MARKET.md](MARKET.md) holds: the recognizer is a
commodity, the durable asset is the **honesty layer** (order, digits, review
queue). Accuracy work should feed that layer, not chase glyph CER.

## 2. Levers, in ROI order

### Lever 1 — shipped 2026-10-01

`--deva-lines auto` is now the default for image inputs (CLI + web studio);
PDF inputs keep the engine reading and the library default stays `off`.
Recorded gate (Appendix AE): letterpress CER **0.434 → 0.373** (42/69 pages
engaged) at **+0.62 s/page**, with zero engagement and byte-identical output
on the modern scan set. Adversarial gate tests added: saturated-but-bright
paper, neutral shadow, dark neutral paper, geometry veto on aged paper.

### Lever 2 — digit trust: N5 first, then N3 (the moat; 1–2 weeks)

The queue is the product. On modern PDFs it surfaces only **0.38** of wrong
digits, and whole-page digit sequences are exact on 3/41 pages. The CRNN
disagreement verifier was stopped at precision 0.50 (Appendix AC) with an
explicit note: *needs a better model or more real ground truth*. Both are
actionable:

1. **N5 — real digit ground truth.** The tooling exists:
   `scripts/build_digit_crop_sheet.py` cuts digit-dominant crops from
   unlabeled modern pages and Cornell scans, writes a review TSV/HTML, and
   `--import` validates corrections into a recognizer training set
   (`digit_lines_v1`). A few hundred corrected crops is 2–3 hours of human
   work on top of an already-built harness.
2. **Synthetic digit injection** — extend the Qt renderer + degradation mix
   with amounts, dates and case numbers (exact GT, controllable difficulty).
3. **Fine-tune a digit-capable recognizer** (N4 recipe, digit-first): the
   letterpress digits currently come out as Latin insertions; modern PDFs
   fail confidently.
4. **Pre-registered gates** (from N2/AC/N4): verifier precision ≥0.6 →
   flag-only, ≥0.8 → test replacement; product gate: digit queue R@10
   **+≥3 pp** on both frozen domains at **≤ +1 s/page**, invented tokens ≈0.

Expected: PDF queue 0.38 → 0.5+, digit-exact pages up from 3/41, letterpress
0.88 held. This is the highest-value accuracy work for the product's claim.

**Status 2026-10-01 — attempted, stopped.** N5 was executed first (219
human-corrected crops, `xheavy`-augmented, Kaggle warm-start fine-tune from
the shipped checkpoint). Verifier precision on the frozen modern pages stayed
**0.0** (145 conflicts, 0 fixes) — the pre-registered `<0.4 → stop` rule
applies; letterpress moved +1.07 pp digit-exact inside the CI while CER/bagCER
moved marginally worse, so the shipped reader is unchanged. Full numbers:
[PLAN.md](PLAN.md) Appendix AF. Remaining paths unchanged: target-domain
ground truth at an order-of-magnitude larger scale, the opt-in bodhan verifier
(precision 0.81, +8.3 s/page), or re-testing a stronger recognizer class
through the same gates.

### Lever 3 — the structure gap on hard tables (1–2 weeks)

The remaining pages (`supreme_218512` court registers) expose **every** arm,
including the VLMs — and the VLMs' advantage is mostly ordering. bagCER says
we already match them on glyphs. So: better structure, not a bigger model.

Do: cell-aware grouping for register grids (label/value columns keyed to
ruling lines), validated on the hard-10 + full 41. Gate: hard-10 CER −≥15%
relative, byte-identical elsewhere, invented unchanged.

### Lever 4 — degradation and real photos (weeks; needs data)

All photo numbers are a synthetic proxy. The path:
1. Stand up `nepali_photo_real_v1`: a consented, anonymized real phone-photo
   set (community collection; even 30–50 pages beats a proxy).
2. Targeted restore experiments (dewarp, shadow, denoise) — each must pass
   the W2.1-style gate that killed sauvola (invented tokens ≈0, no CER
   regression).
3. If photos become line-like after restore, extend the degradation mix used
   to train the in-repo reader.

This is the biggest user-facing gap (0.370/0.757) and the one that most needs
real ground truth rather than more modeling.

### Lever 5 — letterpress recognition ceiling (research, lower priority)

Appendix N1: segmentation + missing dominate the letterpress taxonomy (line
assembly and detection), not recognition. The four W1 training attempts show
diminishing returns from more training alone. Next gains likely come from
line split/merge logic, not a bigger CRNN.

### Explicitly not now

- **VLM full-reader** — fails cost (152 s/page), invented tokens (224), and
  Devanagari numerals come out Bengali (digit-exact 0.077). A hybrid
  (VLM only for low-structure-confidence pages, GPU-optional) stays Phase 5,
  gated on the same honesty invariants.
- **PaddleOCR-VL** — no Devanagari evidence yet; re-run the existing
  bake-off harness when it has one.
- **Surya 2 as default** — best median, but OpenRAIL-M weights, 172 s/page,
  82 invented tokens.

## 3. Guardrails (unchanged, non-negotiable)

- Pre-registered adopt-if gates; frozen sets never used for tuning; a failed
  attempt ends in a recorded stop, not a rewrite of the claim.
- Honesty invariants: invented tokens ≈0; the dual-stream audit stays; the
  queue may never silently replace text.
- Cost: ≤ +1 s/page for recognition changes; CPU-first default; heavy models
  opt-in only.
- Every adopted change updates README / EVALUATION numbers and the public
  benchmark — which doubles as the citable AEO asset ([SEO_AEO.md](SEO_AEO.md) §5).

## 4. Immediate checklist

| # | Step | Lever | Effort | Needs |
|---|---|---|---|---|
| 1 | Adversarial gate tests + image default `auto` — **shipped 2026-10-01** (Appendix AE) | 1 | done | — |
| 2 | Build + label the digit crop sheet, `--import` | 2 | ~1 day | human labeling |
| 3 | Synthetic digit injection in the renderer mix | 2 | days | — |
| 4 | Digit recognizer fine-tune (Kaggle) + run gates — **attempted 2026-10-01, stopped** (Appendix AF: precision 0.0) | 2 | done | #2, #3 |
| 5 | Table cell-grouping spike on the exposed pages | 3 | days | — |
| 6 | Real-photo collection protocol draft | 4 | days | community |

Each attempt lands as its own appendix (R-/N-series style) with numbers —
or it does not ship.
