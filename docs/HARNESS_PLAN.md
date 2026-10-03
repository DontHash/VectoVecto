# Harness plan — multi-read consensus, constraint reconciliation, correction memory

Status: **design + pre-registered gates only** (no implementation). Planning date
2026-10-02. Extends [ROADMAP.md](ROADMAP.md) Tracks 1–2 after the AG/AH failures
and the AI correction-loop beta. Each track ships only through its own gate; a
failed gate means *record, keep opt-in reference, stop* — frozen sets are never
re-tuned.

Thesis: the residual is **recognition inside correctly boxed cells** (Appendix
AH: 3 of 156 v2 digit errors are provably impossible dates; valid-looking
substitutions bypass single-token rules). The next value is not a bigger model —
it is a harness that (A) reads suspect tokens more than once, (B) uses page
context to choose among candidates, and (D) reuses what humans already decided.
A produces candidates, B scores them, D remembers the answers; the studio
renders one suggestion channel.

## 0. Evidence base (why these three)

| Fact | Source |
|---|---|
| bagCER tied across all engines (0.360–0.400): glyph recognition is a commodity | README, BENCHMARK |
| Harness moves worked: raw → pipeline 0.530 → 0.255 CER; repass 0.15 → 0.38 digit R@10 | EVALUATION, PLAN Q/W0.3 |
| Model/rule attempts failed: splitter (AG), validator (AH) | PLAN Appendixes AG/AH |
| Correction loop v1: 99/99 tokens acted, 46 audited digit labels, no consumer yet | PLAN Appendix AI, CORRECTIONS |
| `Token.alt_text` is a single last-writer-wins slot; `flags` can duplicate; calibration is not used in ranking | ocr.py:79, 861–875, 1044–1056 |

## 0b. Step-0 baselines (reproduced 2026-10-03)

All four frozen manifests verified by content hash before scoring
(`eval_freeze.py --check`). Machine: this workstation, RapidOCR on DirectML
(~0.8 s/page OCR). Raw JSON: `out/harness_baselines/` (gitignored); these are
the reference numbers steps 1–2 gates compare against.

| Set | Harness | Key numbers |
|---|---|---|
| `nepali_pdf_v2` hard-10 | `eval_flags --repass-digits --limit 10` | digit R@5 **0.3043** / R@10 **0.3478**; P@10 0.1250; 190 digit tokens |
| `nepali_pdf_v2` full-41 | `eval_flags --repass-digits` | digit R@5 0.3333 / R@10 **0.3846**; P@10 0.2256; **156** digit errors (0.3164 of 493 digit tokens) |
| `nepali_pdf_v2` hard-10 | `eval_document --pages 10 --methods pipeline` | CER **0.2550**, bagCER **0.3758**, digit coverage 0.2857, digit FA 0.3805, invented **665** (strict GT rule), 6.50 s/page |
| `heidata_printed` full-69 | `eval_flags --repass-digits` | digit R@5 0.6788 / R@10 **0.8832**; P@10 **0.1754** |
| `cornell_real` full-16 | `eval_date_flags --role other` (AH convention) | digit R@10 **0.2472**, P@10 **0.2588**, 89 digit errors |

All numbers match the published appendices (AG hard-10 0.3478, AH v2 0.3846,
cornell 0.2472/0.2588, letterpress 0.8832/0.1754, README hard-10 0.2550/0.3758),
so the gate baselines are trustworthy on this machine.

**Two conventions the new gate scripts must mirror** (both bit us in step 0):

1. **P@10 denominator.** `eval_flags.py` accumulates top-K only on pages with
   digit errors (cornell P@10 = 0.3235 there); `eval_number_split._queue_metrics`
   — the AG/AH gate convention — sums top-K over **all** pages (cornell P@10 =
   0.2588). The new gates must reuse `_queue_metrics` semantics so their P@10
   clauses are comparable to the published baselines.
2. **Cornell language.** The AH non-target clause ran cornell with the script
   default `--lang ne`; with the sensible `--lang en` the numbers differ
   (R@10 0.40). Non-target clauses must state the language they ran.

Concrete quantities for the B reachability disclosure: v2 has **156** digit
errors over 493 digit tokens (35/41 pages affected).

## 1. Invariants (unchanged, enforced by gates)

1. **Flag/evidence only.** A and B never edit `Token.text`, `conf`, bboxes or
   reading order. `Token.text` changes only through a human correction (D).
2. **No silent suggestion.** A suggestion is never applied automatically; the
   human confirms, and provenance records it.
3. **Text identity.** On sets where a feature does not engage, `.txt`/`.md`
   text output is unchanged; `--identity` arms assert this per page.
4. **Determinism.** Fixed panel transforms, fixed order, no RNG; no timestamps
   in text outputs.
5. **Cost.** ≤ +1 s/page for recognition changes (A), ≤ +0.1 s/page for
   post-OCR flag work (B). D is offline-inert.
6. **Additive schema only.** Existing `ocr.json` / `corrected.json` consumers
   (`webapp/vvweb/pipeline.py:103–153`, `corrections.py:233–259`) keep reading
   the same keys; new keys are optional.
7. **Pre-registered gates.** Off/on arms on frozen manifests verified by hash;
   dev tuning only on `heidata_dev_v1`; PASS = all clauses true.

## 2. The unified contract (integration spine)

All three tracks meet in two optional, additive `Token` fields (serialized only
when non-empty), plus one existing field left untouched:

```jsonc
// per token (ocr.json "tokens[]"), all optional
"reads": [                                  // A: every independent re-read
  {"source": "repass", "text": "२०८१", "conf": 96.1},
  {"source": "panel:rot+1.5", "text": "२०८१", "conf": 91.4},
  {"source": "panel:otsu", "text": "२०८०", "conf": 88.0}
],
"suggestions": [                            // B/D: candidate labels for humans
  {"text": "२०८१-०४-३२", "source": "context",
   "why": "date column: 12/14 siblings read २०८१"},
  {"text": "२०८१-०४-३२", "source": "memory",
   "why": "corrected 3× before"}
]
```

Rules:

* `alt_text` keeps its current meaning and writers (dual-stream, verifier) — it
  is **not** a third writer target and existing UI keeps working. The studio
  may merge it into the suggestion chips it already renders.
* Review rows in `ocr.json` gain `suggestions` (additive); the studio review
  mode renders chips from it. `three_way_agreement` (metrics.py:287) is the
  precedent for comparing readings; `_digits_of` (ocr.py:1066) is the
  comparison key for digit strings.
* Source labels are stable: `repass`, `panel:<transform>`, `audit`, `verifier`,
  `context`, `memory`, `human`.

## 3. A — multi-read consensus on flagged tokens

**What changed vs today.** Today one re-read exists: the 2× digit re-pass on
digit tokens with `conf < 95` (max 40/page), recorded in `repass_text` and
promoted to `digit_conflict` only on disagreement. A extends this into a small,
deterministic **panel**: for already-flagged digit tokens, re-read the same crop
under 2–3 transforms and record every reading. Agreement becomes a precision
signal; disagreement becomes a candidate and a flag.

**Panel (v1, CPU, no new dependencies).**

| Reader | Transform | Cost |
|---|---|---|
| `repass` | existing 2× Lanczos crop (reuse if computed) | existing |
| `panel:rot+1.5` | 2× crop, +1.5° rotation | ~1 crop read |
| `panel:otsu` | 2× crop, Otsu/adaptive threshold | ~1 crop read |

Tesseract (the other installed backend) is deliberately **not** in v1; it can
join as a 4th panel read only if v1 plateaus (same gate, re-run).

**Placement.** `ocr_page(..., multi_read="off" | "panel")`, after
`apply_digit_repass` (ocr.py:508–512); pipeline forwards `multi_read` to the
**primary pass only** (audit pass `off`), so no double cost. Library default
`off`; CLI/web default stays `off` until the gate passes (mirrors AG/AH).
Suspects = flagged digit tokens, aspect ≤ 8 (verifier rule, ocr.py:1019–1022),
riskiest-first, cap 12/page (verifier precedent, ocr.py:1001).

**Signals.**

* `consensus` derived per token: `agree` (all digit strings equal),
  `split` (≥2 distinct), `unreadable` (no digits from any read).
* New flag `multi_read_conflict` (weight dev-tuned, initial 2.5, vs
  `digit_conflict` 3.0 / `digit_uncertain` 1.5) when a **non-repass** panel read
  disagrees with `text` and the token is not already `digit_conflict`.
  Append only if absent (existing code does not de-duplicate flags — A must).
* Every disagreeing reading is a `suggestions` entry (`source: panel:*`), never
  `alt_text`, never `text`.
* `token_risk` gets no new terms in v1; the flag weight carries the signal.
  If P@10 displacement appears, cut the weight on dev before touching ranking.

**Pre-registered gate — `scripts/eval_multi_read.py`** (mirrors
`eval_number_split.py` shape: off/on arms, `clauses`, `pass`):

| Clause | Bar |
|---|---|
| `digit_r10_plus3pp` on `nepali_pdf_v2` (full 41) | R@10 ≥ baseline + 0.03 |
| `hard10_cer_not_worse` | ≤ 0 (text is never edited; expect exact) |
| `text_identical` all sets | 0 changed pages |
| `invented_not_increased` (`nepali_pdf_v2`) | Δ ≤ 0 (state the rule: strict GT-only, per AG) |
| `flag_precision_ge_0.5` for `multi_read_conflict` on v2 | precision ≥ 0.5 |
| `r10_p10_not_worse` on `heidata_printed`, `cornell_real` | Δ ≥ 0 for both |
| `queue_not_larger` on letterpress + cornell | flagged tokens Δ ≤ 0 |
| `latency_ok` | Δ ≤ +1.0 s/page |

Decision: all pass → adopt (policy default mirrors repass: ON for Devanagari
images, PDFs keep their current policy); any fail → opt-in reference only,
record in PLAN appendix, stop.

**Status 2026-10-03: attempted — gate FAILED** (PLAN.md Appendix AJ). v2 digit
R@10 0.3478 → 0.3478 (hard-10) and 0.3846 → 0.3846 (full-41); new-flag
precision 0.00 / 0.20 vs the 0.5 bar; identity, CER, invented and latency
clauses passed; cornell non-target clauses passed. Diagnosis: 9/13 splits were
tokens already `digit_conflict`, and the new disagreements were transform
artifacts (mostly Otsu) — the "any single read disagrees" rule is too weak and
the suspects were already-flagged tokens. `multi_read="panel"` ships as
opt-in reference only. Redesign direction for a future attempt: ≥ 2 disagreeing
reads (Otsu recorded but not flag-bearing) and wider suspects, tuned on
`heidata_dev_v1`.

## 4. B — page-level constraint reconciliation

**What changed vs today.** AH proved single-token format rules cannot see
valid-looking substitutions (`३०` vs `३१`). B does not try to see them either;
it uses **other tokens on the page** as evidence. Model proposes (candidates
from A, repass, audit, verifier), harness disposes (page-local constraints,
flag-only).

**Candidates.** The distinct digit strings among `text`, `repass_text`,
`alt_text`, `reads[].text`, deduplicated. Tokens with one candidate are
untouched. Date-shaped tokens additionally get parsed components
(`_DATE_SEP_RE` / `_DATE_BARE_RE`, ocr.py:928–932).

**Constraints (v1, page-local, no external tables):**

1. **Date-column agreement.** Cluster date-shaped tokens by x-overlap
   (reuse the column primitives in `layout.py`: `_find_gutter`, `_grid_columns`
   are module-level and take token sequences). With ≥ 4 siblings in a cluster,
   if ≥ 70% share a year/month component and exactly one token's candidate set
   contains a reading matching the majority while `text` does not → suggest it,
   flag `context_conflict`.
2. **Format consistency.** Within a cluster, separator/script must match the
   majority (`-` vs `/` vs `।`); a candidate restoring the majority format is
   preferred.
3. **Prefix groups.** Tokens sharing a non-digit prefix pattern (case numbers,
   file numbers, e.g. `०७६-०२५३-*`): a candidate matching the cluster prefix
   while `text` deviates is preferred.
4. **In-page repeat.** If `text` occurs elsewhere on the page but a candidate
   is the repeated reading, prefer the repeat (weak; tie-breaker only).

Sums/checksum constraints (visible table totals) are explicitly **v2** — they
need table-total detection and are riskier than the four above.

**Placement.** New module `veriscript/document/reconcile.py`; called in
`run_document_pipeline` after reading order (bboxes are preserved;
`sort_reading_order` is order-only) and before export; `reconcile=False` param
on the pipeline. Pure CPU post-processing, no image access.

**Signals.** Flag `context_conflict` (weight dev-tuned, initial 2.0, below
`script_mismatch` 2.5) plus a `suggestions` entry with a human-readable `why`.
If two candidates tie → no flag (ambiguity is not evidence). Never edits text,
never writes `alt_text`.

**Pre-registered gate — `scripts/eval_context_reconcile.py`:**

Before the main gate, the script must print the **reachability disclosure**
(AG/AH lesson): among the v2 digit errors, how many have ≥ 2 candidates and a
context signal that prefers the ground-truth reading. If reachable < 10, the
gate is expected to fail on arithmetic and the run is still recorded.

| Clause | Bar |
|---|---|
| `digit_r10_plus3pp` on `nepali_pdf_v2` (full 41) | R@10 ≥ baseline + 0.03 |
| `flag_precision_ge_0.5` for `context_conflict` on v2 | ≥ 0.5 |
| `text_identical` all sets | 0 changed pages |
| `r10_p10_not_worse` on `heidata_printed`, `cornell_real` | Δ ≥ 0 |
| `queue_not_larger` on letterpress + cornell | Δ ≤ 0 |
| `latency_ok` | Δ ≤ +0.1 s/page |

Run the gate twice: with A `off`, then with A `panel` on, to show whether B
composes with A. Adopt policy identical to A.

## 5. D — correction memory

D has no frozen-set gate (it is **offline-inert**: no memory ships in frozen
data, so offline numbers cannot change). Its invariants and product metrics are
the gate. Two levels, shipped in order:

**D1 — in-run repeats (no persistence, no consent needed).**

* Studio groups flagged tokens with identical normalized reading + flag
  signature; confirming/correcting one offers "apply to N identical".
* The API needs no change (the UI sends N corrections; `corrections.py`
  index matching already handles batches).
* Value: registers repeat dates/case numbers constantly; the beta showed a
  99-token queue is mostly repeats.

**D2 — local correction memory (opt-in, local, text-only v1).**

* New module `veriscript/document/memory.py`; JSONL store under
  `VERISCRIPT_MEMORY_DIR` (default `~/.veriscript/memory.jsonl`), **disabled by
  default**; hosted multi-tenant deployments keep it off (no cross-visitor
  leakage without accounts).
* Record on accepted human action: `{created, engine, lang, original,
  corrected, action, flags, n_digits, value_shape}`. **No crops, no page
  images, no run ids** in v1 — text only, consistent with the privacy posture
  (crop/phash matching is a later opt-in if demand is proven).
* Suggestion match: exact `(normalized original, flags signature)` first; else
  same digit-length + Hamming ≤ 1 + same signature. Suggestions surface as
  chips (`source: memory`) and are never auto-applied.
* Lifecycle: accepted/rejected counters per record; suppress below an
  acceptance floor; a "clear memory" UI action and endpoint; the store is
  outside `PUBLIC_FILES` and never served; documented in CORRECTIONS.md.
* Product metrics (10 beta sessions, non-frozen pages): suggestion acceptance
  ≥ 60%, acted-on ≥ 20% (already exceeded in AI), median seconds/token down
  vs the AI session; the queue must still empty.

**Known dependency:** web `/correct` handles page 1 only (CORRECTIONS.md
non-goal); D1 across PDF pages waits on multi-page correction plumbing. CLI
parity for memory is deferred.

## 6. Sequencing

| Step | Work | Size | Gate / exit |
|---|---|---|---|
| 0 | Contract fields (`reads`, `suggestions`) + export/review plumbing + tests; reproduce frozen baselines (§0b). Gate scripts land with their features (steps 1–2) so they can never score a nonexistent flag. | **done 2026-10-03** | baselines match EVALUATION ✓ |
| 1 | A: panel + `multi_read_conflict` + `scripts/eval_multi_read.py` | 3–5 d | **done 2026-10-03 — gate FAILED** (Appendix AJ); opt-in reference only |
| 2 | B: `reconcile.py` + `scripts/eval_context_reconcile.py` | 4–6 d | B gate, A off and on |
| 3 | D1: repeat grouping + suggestion chips in the studio | 2–3 d | invariants + beta metrics |
| 4 | D2: opt-in local memory + lifecycle + clear + docs | 3–4 d | product metrics |
| 5 | Republish BENCHMARK/README numbers for whatever passed; record failures as appendices | 1 d | docs updated |

Failures at step 1/2 do not block D (independent), and B can run without A
(baseline candidates already come from repass/audit/verifier).

## 7. Non-degradation matrix

| Change | Text outputs | Queue / R@10 | ocr.json | Perf | Default |
|---|---|---|---|---|---|
| A panel | never edits | new `multi_read_conflict` flag + suggestions; gated | `reads`/`suggestions` additive | ≤ +1 s/page, cap 12 tokens | off until gate |
| B reconcile | never edits | new `context_conflict` flag + suggestions; gated | `suggestions` additive | ≤ +0.1 s/page | off until gate |
| D1 repeats | only via human action | web-only grouping; offline unchanged | unchanged | negligible | always on (UI) |
| D2 memory | only via human action | offline unchanged (empty memory) | `suggestions` additive | negligible | off (opt-in) |

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A panel reads correlate (same engine) → little lift | reachability disclosure in the gate; alternate-stream read reserved; tesseract as 4th read only if needed |
| New flag displaces true errors from top-10 (AH cornell P@10 lesson) | flag weight dev-tuned on `heidata_dev_v1`; P@10 clauses on all non-target sets; weight cut or feature stop on fail |
| B has too few reachable errors (AH arithmetic) | disclosure printed before the gate; if < 10 reachable, stop without burning a full run |
| Column heuristics misfire on multi-column pages | require ≥ 4 siblings and ≥ 70% majority; identity clause on letterpress; no flag on ties |
| Suggestion anchoring (human accepts a wrong chip) | never auto-applied; provenance records suggestion acceptance; rejected hits suppress the record |
| Memory privacy expectations | off by default, text-only, outside the servable whitelist, clear action, documented; hosted demo stays off |
| Duplicate flags / `alt_text` clobbering | A/B append unique flags and write suggestions, not `alt_text`; contract tests pin this |

## 9. Explicit non-goals

* No new model, no CRNN fine-tune, no VLM default (AG/AF falsified).
* No text replacement by any harness signal; no auto-accept of suggestions.
* No cross-page constraint search in v1 (D2 memory is the cross-page carrier);
  no checksum/sum constraints in v1.
* No crop or page-image persistence without a new explicit consent step.
* No frozen-set tuning; no metric renegotiation after a run.

## 10. Open questions for the owner

1. If A passes, is the default policy "Devanagari only, mirroring repass", or
   images-only (mirroring `deva-lines auto`)? Proposed: mirror repass.
2. Should tesseract join the panel in v1 when installed, or only on a v2 gate?
   Proposed: v2.
3. Hosted demo keeps D2 off (confirmed by privacy posture)? Proposed: yes —
   D1 only there.
4. Are red/amber overlay semantics for the two new flags decided by the gate
   (`multi_read_conflict` = amber, `context_conflict` = amber until precision
   says red)? Proposed: amber for both, revisit after measurement.
