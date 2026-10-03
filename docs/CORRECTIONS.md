# Correction loop — verify → fix → re-export (design)

Status: **v1 shipped and verified end-to-end** (core + CLI + web API + studio
review mode; the browser walkthrough is recorded at the end of this file).
This is the Track-2 item of [ROADMAP.md](ROADMAP.md); it is product and data
infrastructure, not a frozen-set accuracy change, so no accuracy gate applies —
the success metrics in §8 do.

## 1. Why this exists

The review queue is the product's differentiator, but it was **read-only**: a
user sees "check token ७" and cannot act on it, and their knowledge is
discarded. At the same time, the N5/AG/AH track proved the residual is
*recognition inside correctly boxed cells* — precisely the knowledge only a
human reader can supply. The loop makes the honesty contract complete
(verify → fix → re-export) and turns normal use into **consented**
target-domain ground truth, with no telemetry and nothing leaving the machine
unless the user explicitly exports it.

## 2. The loop

```
upload ──► OCR + review queue ──► review mode (fix / confirm flagged tokens)
                                        │
                                        ├─► corrected artifacts (PDF/TXT/MD/JSON + provenance)
                                        └─► explicit export of marked crops + labels ──► training data
```

Every step is local. Corrections are applied in the run directory; the
optional export is a file the user creates and downloads themselves.

## 3. Data contract

### 3.1 Corrections file (`veriscript.corrections` v1)

```json
{
  "kind": "veriscript.corrections",
  "version": 1,
  "created": "2026-10-02T12:00:00Z",
  "run_id": "ab12cd34ef567890",
  "pages": {
    "page_ab12cd34": [
      {
        "index": 42,
        "bbox": [120, 340, 260, 380],
        "original": "२०८१-०४-३९",
        "corrected": "२०८१-०४-३२",
        "action": "changed"
      }
    ],
    "*": []
  }
}
```

* `index` — position in the page's exported `tokens` array (stable within a
  run); `bbox` is a cross-check.
* `action` — `changed` (default) or `confirmed` (the engine reading is
  accepted; the token still leaves the queue).
* Page key — the artifact stem (`page_x` for images, `page_x_p003` for PDF
  pages); `"*"` means "the page being processed" for single-page files.
* One file per document is the unit; it is hand-editable.

### 3.2 Matching rules (never silently drop a correction)

1. `index` in range and (`original` empty or equals the token text, or the
   bbox is close) → apply.
2. otherwise, the token whose bbox center falls inside `bbox` and whose text
   matches `original` (when given) → apply.
3. otherwise → `skipped`, and the caller reports the count.

No correction ever silently rewrites a different token.

### 3.3 Token provenance

`Token` gains `orig_text` (pre-correction reading, set only when the text
changed) and `corrected_by` (`"human"`). `text_source` becomes `"human"` when
the text changed. Export rules:

* corrected tokens leave the review queue (`flags` cleared) — that is the
  point of reviewing;
* `ocr.json` carries `original_text`, `text_source`, `corrected_by` for every
  corrected token, so the original reading is never lost and the corrected
  record is self-describing;
* the searchable PDF, transcript and markdown all use the corrected text.

### 3.4 Corrected artifacts

Written next to the originals, never overwriting them:

| artifact | name |
|---|---|
| searchable PDF | `corrected.pdf` |
| transcript | `corrected.txt` |
| markdown | `corrected.md` |
| OCR JSON (with provenance) | `corrected.json` |
| applied corrections record | `corrections.json` |

## 4. CLI

```
python cli.py --mode document --input page.pdf --output out \
    --apply-corrections corrections.json
```

Runs the normal pipeline, applies the matching page entries (or `"*"`), and
writes `{stem}_corrected.pdf/.txt/.md/.json` plus `{stem}_corrections.json`,
printing `changed / confirmed / skipped`.

## 5. Web API

* `POST /api/runs/{run_id}/correct` — body `{"corrections": […]}`; applies to
  the served page, writes the corrected artifacts + `corrections.json`, and
  returns the updated review queue, transcript, flags summary and file links.
* `POST /api/runs/{run_id}/corrections/export` — body
  `{"corrections": […], "share": [indices]}`; writes `corrections.zip`
  containing `corrections.json`, `labels.tsv`, `crops/<page>_<index>.png` for
  **only** the marked indices, and a `README.txt`. Returns the download link.
* Both are behind the same auth/artifact whitelist as the rest of the studio;
  inputs are size- and shape-validated (`≤ 500` corrections, text `≤ 200`
  chars). No OCR runs, so no quota is charged.

## 6. Studio review mode

In the queue tab:

* a row shows the engine reading, the kept alternative (when present) and the
  flag chips;
* `Enter` confirms the engine reading, typing replaces it, `Tab`/arrow keys
  move between rows, `Esc` clears an edit;
* a counter shows `reviewed N/M`; `Apply corrections` posts the batch and the
  download row switches to the corrected artifacts;
* `Export corrections for training` presents **unchecked** per-item
  checkboxes; picked items are cropped from the run's restored image and
  packed into the zip. Any item not picked is never sent anywhere.

## 7. Privacy invariants (non-negotiable)

1. No telemetry, no auto-upload, no background contribution.
2. `corrections.zip` is created only by an explicit user action and contains
   only picked items; no full-page images, ever.
3. Corrections live in the run directory under the normal retention TTL unless
   the user downloads them.
4. The UI never implies a correction is shared.

## 8. Success metrics

* e2e (frozen-free): apply → corrected artifacts differ from the originals
  exactly on corrected tokens; provenance present in `corrected.json`.
* Privacy: an export with `share: []` contains zero crops.
* UX: review mode ships with a 10-run beta; `≥20%` of queued tokens acted on
  in a session is the early-usefulness bar. **Measured 2026-10-02** (PLAN.md
  Appendix AI): 10 sessions on real court-register pages, **99/99 queued
  tokens acted** (88 changed / 11 confirmed / 0 skipped), every queue emptied,
  and **46 digit-bearing labels** staged as
  `data/doc_eval/target_domain_digit_staging_v1`. The dataset owner then
  audited all 46 and approved them unchanged — the batch is human-audited
  target-domain digit ground truth (internal, not a frozen eval set).
* Data: exported zips are loadable by the training pipeline
  (`scripts/build_digit_crop_sheet.py --import` after conversion to
  `lines/ + labels.tsv`; the zip already carries `labels.tsv`).

## 9. Non-goals (v1)

* Adding tokens the detector missed (box drawing) — v2.
* Multi-page combined corrected PDFs (page-1/web scope in v1; the CLI covers
  every page of a PDF).
* Cloud sync/accounts. The loop is local-first by design.

## 10. Verified walkthrough (2026-10-02, local studio)

Driven in the browser against `webapp/server.py` (built app, real RapidOCR):

1. `clean-invoice.png` (23 tokens, 14 flagged) → review mode listed 14 rows.
2. First row corrected (`INVOICE` → `INVOICE TEST`) and Enter-staged: the bar
   read "1 staged · 14 still in the queue", the export panel appeared with the
   item unchecked.
3. `apply corrections (1)` → `applied: 1 changed · 0 confirmed · 0 skipped`,
   the queue dropped to 13 rows, and `corrected.pdf/.txt/.json` +
   `corrections.json` links appeared; `corrected.txt` opened with the
   corrected first line.
4. Ticking that one item and building the archive produced a 7.5 KB zip
   (`PK` magic) containing one crop — no page images.

A bug found by this walkthrough and fixed: the run summary originally kept
reading `ocr.json` after corrections, so the queue still showed the reviewed
token; `collect_run_summary` now prefers `corrected.json` / `corrected.txt`
when they exist (the originals stay downloadable). Regression test:
`tests/test_webapp_api.py::test_correct_applies_and_serves_corrected_files`.

## 11. Suggestions and correction memory (track D, docs/HARNESS_PLAN.md §5)

Review rows may carry `suggestions: [{text, source, why}]` — candidate labels
the human can accept with one click (`source` is `consensus`, `context` or
`memory`). Suggestions are **never auto-applied**; accepting one stages a
normal correction, and the correction payload then carries `suggested` +
`suggestion_source` so the outcome can be measured.

**D1 — identical repeats.** Rows whose reading is identical to another row in
the queue show `apply to N identical`, which stages the same fix (or
confirmation) for every copy in one click. No persistence, works on the hosted
demo.

**D2 — local correction memory (opt-in).** `VERISCRIPT_MEMORY=1` enables a
text-only JSONL store (`VERISCRIPT_MEMORY_DIR`, default
`~/.veriscript/memory.jsonl`); it records accepted corrections and
accept/reject feedback. Matching is exact full-reading + flags signature
first, then same skeleton (digits masked) + same digit length + Hamming ≤ 1;
records with net-negative feedback are suppressed. Nothing leaves the machine:
no page images, no crops, no bboxes, no run ids. The studio shows a "local
correction memory is on" line with a **clear memory** action; `GET /api/memory`
reports the count and `POST /api/memory/clear` removes it. Hosted multi-tenant
deployments leave it off (the default), so no visitor's corrections can leak
into another visitor's suggestions.

**Measured (2026-10-03, offline replay of the 10 beta sessions,
`scripts/eval_memory_replay.py`).** Memory from earlier pages, queried on
later ones against the human's actual outcome: **1 suggestion in 91 queries
(1.0% coverage), 1 hit (100% precision when it fires), 1.0% resolved**. The
court-register queue tokens are long, mostly unique lines, so value repetition
does not translate into line-level suggestions; within-page duplicates were
0/99. The mechanics are precise but the value hypothesis at line granularity
is not supported by this corpus — the plausible next direction is a
digit-value-level memory (4 repeated digit values covered 18/99 occurrences),
not more line matching. D2 therefore stays opt-in and unproven; the ≥60%
suggestion-acceptance bar could not be assessed on this data.
