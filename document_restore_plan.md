# VectorScaling — Faithful Document Restore

**Status:** Product plan v2 (researched). Execute in order. Do not skip the eval harness.
**Decision date:** 2026-09-20 (v2 revisions same day)
**Relationship to old roadmap:** This *replaces* photo-SR as the default product. The math roadmap (`math_based_image_upscaling_roadmap.md`) remains a research appendix. Phase 4.5 (vector/raster split) and Phase 8 (reconstruction constraint) are reused, aimed at pages instead of portraits.

**v2 research corrections (verified, see §9a):**
1. `PyMuPDF` is **AGPL-3.0** → banned from the shipped path; searchable PDF is written with **reportlab (BSD-3)**.
2. Tesseract is an external install and is **not present** on the dev machine → **RapidOCR (PP-OCRv6, ONNX, Apache-2.0)** ships as default, Tesseract is an optional second adapter, both measured per language.
3. Per-glyph OCR confidence does not exist in either engine → per-line confidence + **digit re-pass** + (v2) two-pipeline agreement gate.

---

## 1. Product one-liner

Local restore for mixed documents where a wrong character costs money.
Sharp when we are sure. Honest when we are not.

**Job to be done:** Drop a scan, phone photo of a bill, screenshot, or PDF page. Get a cleaner image **and** searchable text, with low-confidence glyphs marked instead of invented.

**Not the job:** 4× cinematic photos, anime, license plates, "looks sharper than Upscayl."

---

## 2. Why this product (novelty × usability)

| Bet | Usable? | Novel? | Verdict |
|---|---|---|---|
| Photo 4× SR | Yes (Upscayl exists) | No | Kill as homepage |
| Anime SVG | Weak | Overclaimed | Kill |
| CCTV plates | Only with a dedicated camera | No | Kill |
| Generic "document upscaler" | Yes | No (Adobe Scan, Lens, DocRes) | Insufficient |
| **Faithful mixed-page restore + OCR confidence** | Yes (bills, contracts, archives, UI) | Yes as a *consumer local* product | **This** |

Novelty is not a new backbone. It is three things composed:

1. **Do not invent glyphs.** Confidence flags + agreement gate. Ambiguous `8`/`B` stays soft and flagged.
2. **Route the page.** Text → restore + OCR. Logos/stamps → vectors. Signatures/photos → freeze (no GAN).
3. **Dual output.** Searchable PDF (the work) + optional SVG of true graphics (the bonus).

Usability is one loop: drop file → preview with warnings → download PDF/text. Local by default (privacy is the reason to use us instead of Lens).

---

## 3. Non-goals (explicit)

- Do not train another Real-ESRGAN clone on DIV2K.
- Do not ship UltraSharp / Remacri / grain as document defaults.
- Do not vectorize letters as the primary representation (traced `A` blobs are not searchable).
- Do not claim forensic plate recovery or "enhance CCTV."
- Do not require cloud APIs for v1.
- Do not optimize PSNR as the north-star metric.
- **Do not ship any AGPL/GPL library in the product path** (license gate, §9a).

Photo engines (`sr_engine.py`, ncnn models, `train_v2.py`) may stay in the repo under **Advanced / Photo**. They are not the product.

---

## 4. Users and slices

**Primary (v1)**

- People digitizing bills, receipts, invoices (amounts must not hallucinate).
- Students / researchers with phone photos of papers and old PDFs.
- Anyone who needs a searchable PDF without uploading to a cloud.

**Secondary (v1.1+)**

- Mixed pages: letterhead + stamp + table + a small photo.
- UI / error-message screenshots (high-contrast type).

**Not a user yet**

- Industrial ALPR.
- Full archive shops that already run ScanTailor + ABBYY (we learn from them; we do not beat them on batch in v1).

---

## 5. Success metrics

North star: **character error rate (CER)** on a frozen eval set, plus **hallucination rate**.

| Metric | Definition | v1 bar |
|---|---|---|
| CER | `edit_distance(ocr, gt) / len(gt)` (jiwer) | Beat raw OCR on the same page by ≥20% relative |
| WER | word-level edit rate | Track; secondary |
| Hallucination rate | restored-OCR tokens absent from GT alignment **and** absent from raw-OCR | **Must not rise** vs raw OCR |
| Digit hallucination | same, restricted to tokens containing digits (amounts/dates) | Must not rise |
| Ambiguity coverage | fraction of true token errors flagged low-confidence | ≥50% |
| False-alarm rate | fraction of flagged tokens that were actually correct | ≤25% |
| Calibration | ECE of line confidence vs correctness (10 bins) | Track; improve over raw |
| Time | page-to-PDF on CPU, 1× A4 ~150–200 DPI | < 8 s median (target <4 s) |
| Privacy | no network on the happy path | Required (air-gap test) |

PSNR/SSIM on glyphs may be logged. They must not decide shipping.

**Kill criteria:** if after Phase D the pipeline does not beat raw OCR + a Sauvola baseline on synthetic pages *and* at least one public real-receipt set, stop adding models and fix restore/OCR config first.

---

## 6. Eval harness first (do not skip)

### 6.1 Datasets — zero human transcription

| Source | GT | Generation cost | Committed? |
|---|---|---|---|
| Synthetic invoices (our renderer) | Exact strings we draw | Free, seeded | Yes (`data/doc_eval/synthetic`) |
| Open PDFs rendered via pypdfium2 | **Text layer** extracted from the PDF | Free, download once | GT + manifest yes, PDFs no |
| Demo PDFs (reportlab-generated round-trip) | Text layer | Free, offline | Test fixture only |
| Public real sets (SROIE / CORD / FUNSD / DocVQA) | Existing annotations | Download | Optional, sanity checks only; never redistributed |

Normalization rules for GT and hypotheses: CRLF→LF, collapse runs of spaces/tabs, strip. Keep case and punctuation (CER is meaningful). Multi-column PDFs may extract in non-reading order — record the clean-render→text-layer **CER floor** for each PDF page and report every method relative to that floor so extraction order noise affects all methods equally.

### 6.2 Scripts

| File | Role |
|---|---|
| `degradation_document.py` | Deterministic document degradations (seeded): JPEG 30–70, uneven illumination + shadow, motion blur, perspective, downscale/re-up, speckle |
| `doc_data.py` | Synthetic invoice renderer, PDF→page GT builder, dataset manifests, demo PDF generator |
| `document_ocr.py` | OCR adapter registry: RapidOCR (default) + Tesseract (optional) → tokens with conf/bbox |
| `doc_metrics.py` | CER/WER, hallucination, digit hallucination, coverage, false-alarm, ECE |
| `eval_document.py` | Restore/baseline → OCR → metrics → JSON report |
| `tests/test_document_eval_smoke.py` | Unit + smoke tests on synthetic "INVOICE 1200.00" |

Baselines in the harness, always:

1. Raw image + OCR
2. Sauvola threshold (`skimage`) + OCR
3. Lanczos 2× + OCR
4. `SmartUpscaler(mode="auto")` + OCR (proves photo SR *hurts* or does nothing; opt-in flag)
5. Our document pipeline (the thing we ship)
6. Optional external: OCRmyPDF (dev machine only, not a dependency)

Ship a JSON report like `out/doc_smoke.json`.

---

## 7. Target architecture

```
PDF / image
    │
    ▼
ingest (pypdfium2 render @150–300 DPI, EXIF rotate)
    │
    ▼
page router (v2; v1 = treat whole page as text)
    ├─ text / table regions     → restore → OCR
    ├─ logo / stamp / badge     → existing Bézier vector (no OCR)
    ├─ signature / photo / stamp ink texture → freeze (identity or mild denoise)
    └─ background paper         → flatten illumination only
    │
    ▼
confidence audit (per-line OCR conf, digit re-pass, v2: recon disagreement)
    │
    ▼
outputs
    ├─ restored PNG
    ├─ searchable PDF (image + invisible text by bbox; reportlab)
    ├─ .txt / .json (tokens, boxes, conf)
    ├─ overlay PNG (low-conf glyphs highlighted)
    └─ optional SVG (graphics only, not the alphabet)
```

**Invariant:** `numpy array in → numpy array out` for the restore step, so the old harness pattern still works. OCR and PDF are wrappers around that.

**Do-not-hallucinate gate (v2, designed in v1):**

- Run OCR on restored *and* on a conservative binarization of the original.
- If a high-stakes token (digit run, amount, date) disagrees, **do not pick the prettier one**. Flag both and keep the original pixels in that box (or show a warning).
- Never let a GAN/SR model be the only voter on digits.

---

## 8. What happens to the current codebase

| Current piece | Fate |
|---|---|
| `app.py` Gradio slider | Keep. Default mode `document`. Photo models → Advanced accordion. Grain default 0 (already). |
| `smart_upscaler.py` | Add `mode="document"` (and treat `auto` as document if page-like). Photo path remains `mode="photo"`. |
| `vector_raster_hybrid.py` | Keep. v2: run on *non-text* masks only (logos/stamps). Do not vectorize glyph interiors in v1. |
| `sr_engine.py` / x4plus / x4v3 | Advanced photo only. Not called from document default. |
| `deep_unfolding.py` fidelity | Candidate *inner* prior later if CER plateaus; not v1. |
| `eval_harness_v2.py` | Keep for photo regression; not the product scoreboard. |
| `train_v2.py` / DIV2K | Frozen unless Phase F opens. |
| `cli.py` | Add `--mode document`, PDF in/out, `--ocr`, `--lang`. |
| `test_smart_upscaler.py` | Keep photo/vector tests. Add document tests; do not break PrakashJI skin gates for `mode=photo`. |
| Examples De1 / PrakashJI | Move to "Photo examples". Document examples: synthetic invoice + one public scan. |

---

## 9. Technical choices (v1 defaults)

| Concern | v1 default | Why | Revisit |
|---|---|---|---|
| OCR engine | **RapidOCR 3.9.x (PP-OCRv6 ONNX), Apache-2.0, models bundled in the wheel (~31 MB)** | Zero external install, offline, per-line conf + boxes, DML/CUDA optional | Tesseract adapter measured per language |
| OCR engine #2 | Tesseract 5 via subprocess (`tesseract --tsv`), optional | Comparison baseline, extra languages, OSD | Promote if it wins a language on the harness |
| PDF read | `pypdfium2` | BSD-3/Apache, fast render, text-layer GT extraction | — |
| Searchable PDF write | **`reportlab` (BSD-3)**, invisible text (`setTextRenderMode(3)`) | Permissive; no AGPL hazard | `pikepdf` (MPL-2.0) if we must edit existing PDFs |
| Restore | Classical OpenCV + skimage: rotate, illumination flatten (morphological background division), light edge-preserving denoise, Sauvola/CLAHE as OCR side stream | Matches ScanTailor's actual gains; no training | DocRes weights only if harness says we lose on dewarp/shadow |
| Text SR / heavy models | Not in v1 | Text SR hallucinates digits | Phase F, behind the same API, MIT DocRes |
| Languages | `ch_en` RapidOCR default; Tesseract `eng` optional | Both offline | `nep`/`hin` as v2 flag (extra model download, then vendored) |
| Scale | Restore at native / 2× max | 4× is rarely the OCR bottleneck | User toggle |
| UI | Existing Gradio | Already shipped | — |

**Display vs OCR:** show a cleaned *grayscale* page (readable). Feed OCR a *binarized or contrast-normalized* sibling. Do not force the user to look at a harsh binary unless they opt in.

### 9a. Verified stack & licenses (shipped path = all permissive)

| Component | Version installed | License | Notes |
|---|---|---|---|
| RapidOCR | 3.9.2 | Apache-2.0 | PP-OCRv6 det/rec/cls ONNX bundled in package |
| onnxruntime (+directml) | 1.20.1 / 1.24.4 | MIT | CPU default; DML optional on RTX 2050 |
| pypdfium2 | 5.10.1 | BSD-3 / Apache-2.0 | Render + text layer |
| reportlab | 4.5.1 | BSD-3 | Searchable PDF writer |
| OpenCV contrib | 4.13.0 | Apache-2.0 | classical restore |
| scikit-image | 0.26.0 | BSD-3 | Sauvola/Niblack |
| jiwer | 4.0.0 | Apache-2.0 | CER/WER |
| Tesseract (optional) | not installed | Apache-2.0 | dev/baseline only |
| DocRes (Phase F) | — | MIT | weights via OneDrive; mirror to GCS if adopted |

**Banned from shipped path:** PyMuPDF (AGPL-3.0), Ghostscript (AGPL), any GPL OCR/model.

**Air-gap requirement:** full pipeline must run with networking disabled (CI test). Pin `rapidocr==3.9.*` and vendor models so future releases never silently download.

---

## 10. Phases

### Phase A — Harness (2–3 days)

**Build:** `degradation_document.py`, `doc_data.py` (synthetic invoice + PDF text-layer GT), `document_ocr.py` (both adapters), `doc_metrics.py`, `eval_document.py` with baselines 1–4, smoke tests.

**Done when:** `python eval_document.py --json out/doc_smoke.json` runs on CPU; JSON includes CER + hallucination for raw/sauvola/lanczos; a written note records photo-SR CER vs raw (expected: not better).

### Phase B — Classical restore core (3–5 days)

**New module:** `document_restore.py`

```python
restore_document(img_bgr, *, scale=1) -> dict
# keys: display_bgr, ocr_bgr, debug (illum, binary, skew_angle)
```

Steps (each function independently testable):

1. EXIF/array orientation; optional 90/180 via OCR cls/OSD later
2. Downscale huge scans to max side 2500 for speed (record scale)
3. Estimate skew (min-area rect of text-like edges or projection profile); rotate
4. Illumination flatten (large-kernel median / morphological closing)
5. Light denoise (edge-preserving)
6. Build `ocr_bgr`: Sauvola or Wolf binary, or CLAHE+binary
7. Build `display_bgr`: illumination-corrected grayscale, *not* over-binarized
8. If `scale==2`, Lanczos on display; OCR on the 2× binary

**Done when:** CER on synthetic degraded invoices beats raw and Lanczos; hallucination does not increase; unit tests (known 5° skew, known shadow gradient) pass.

**Do not** call `sr_engine` here.

### Phase C — OCR confidence + files + CLI (3–4 days)

- `document_ocr.py`: `ocr_page(ocr_bgr, backend="rapidocr", lang=...) -> list[Token]` with `text, conf, bbox`; flag `conf < 60` (tune on harness)
- Digit-run detector: tokens matching `[0-9]{2,}` get a **recognition-only second pass** on an upscaled crop; disagreement → `flag="digit_conflict"`
- `document_export.py`: searchable PDF (reportlab), overlay PNG (green high conf, amber low, red conflict), transcript.txt, ocr.json
- CLI: `python cli.py --mode document --input scan.jpg --output out_dir --ocr --pdf`

**Done when:** PDF text is selectable and roughly aligned; overlay exists; our pipeline is baseline 5 in the harness and wins on synthetic CER.

### Phase D — App default (2–3 days)

- `app.py`: default routing **Document (recommended)**; image *and* PDF (first page in v1); outputs slider + overlay + transcript + files; SVG checkbox off by default; grain slider removed from document pane; photo engines under Advanced
- `smart_upscaler.py`: `mode="document"`, `mode="auto"` page classifier (text-line area + saturation + skin; bias → document)
- Tests: `test_web_app.py` passes for synthetic text; new invoice → overlay + pdf path test

### Phase E — Mixed-page router (the wedge) (week 3–4)

Text boxes (from the OCR det pass) → restore+OCR; graphic blobs → existing Bézier export; signature/photo boxes → copy pixels (no SR); paper → flatten; recompose; amount gate on digit conflicts. Test in `tests/test_document_router.py`.

### Phase F — Only if CER plateaus (optional research)

- DocRes (MIT) **weights as a module** behind the same API; keep the gate
- Small text-SR (TSRN-class) **inside text boxes only**, reconstruction-constrained
- If a borrowed model lowers CER without raising hallucination, **use it** — do not retrain a photo GAN to "be DocRes"

### Phase G — Product polish

Multi-page PDF (v1.1), language packs UI, batch folder in CLI, "I'm not sure" summary of flagged amounts/dates, Windows packaging.

---

## 11. UX copy (ship this tone)

- Title: **VectorScaling — Document Restore**
- Subtitle: Local. Searchable. Won't invent the numbers on your bill.
- Primary button: **Restore & read**
- Status line: `n low-confidence glyphs · k digit conflicts · t seconds`
- If digit conflict: show original crop and restored crop side by side, neither auto-picked.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RapidOCR API/model drift | Pin `rapidocr==3.9.*`, vendor ONNX models, single adapter file |
| Offline guarantee regresses | Air-gap CI test; models vendored |
| Tesseract install hell on Windows | It is optional now; `winget install -e --id tesseract-ocr.tesseract` documented; tests skip when absent |
| Classical restore fails on heavy curl/dewarp | Phase F DocRes dewarp only; do not block v1 |
| Users still judge "prettiness" | Overlay + transcript make OCR the visible product |
| Photo users feel abandoned | Advanced → Photo mode, old engines intact |
| Synthetic/PDF GT skews to clean typography | Photo-realistic print simulation; public receipt sets as real-world sanity check |
| Multi-column PDF text order inflates CER | Report the extraction floor; compare methods relative to it |
| Scope creep (plates, anime) | This document's non-goals; refuse in UI |

---

## 13. Suggested calendar

| Week | Phase | Outcome |
|---|---|---|
| 1 | A + start B | CER harness + first restore beating raw on synthetic |
| 2 | B finish + C | CLI PDF/OCR/overlay |
| 3 | D | App default is document |
| 4 | E | Mixed-page demo (the wedge) |
| later | F–G | Only if numbers demand it |

---

## 14. Distribution & packaging

- **Open core:** CLI + Gradio OSS (this repo), Apache-2.0-friendly dependency set.
- **Paid Pro desktop:** signed portable Windows build (PyInstaller is fine for proprietary apps), batch mode, multi-language packs, priority updates. No cloud requirement, ever, for the happy path.
- Bundle the RapidOCR ONNX models and reportlab fonts; attribution in `MODEL_LICENSES.md`.
- Naming: repo is `VectoVecto`, UI says VectorScaling — pick one brand before Pro launch (open decision).

---

## 15. Open questions (locked defaults)

| Question | Default unless we revisit |
|---|---|
| OCR engine | RapidOCR default; Tesseract optional; harness decides per language |
| Binarize for display? | No; grayscale display, binary for OCR |
| Default scale | 1× restore; 2× optional |
| `auto` means | Document if page-like, else photo |
| SVG default | Off |
| Languages | English/Chinese first |
| Cloud | Never in v1 |
| v1 scope | Single page; multi-page = v1.1 |

---

## 16. Definition of "we shipped the product"

A stranger can:

1. `pip install -r requirements.txt` (no external binaries required)
2. Open the Gradio app
3. Drop a phone photo of a printed invoice
4. Download a searchable PDF
5. See any shaky totals highlighted instead of silently "fixed"

And on the frozen harness, document mode has **lower CER than raw OCR and lower CER than photo SmartUpscaler, with hallucination rate no worse than raw**, while the pipeline runs with networking disabled.

---

## 17. First implementation slice (one PR series)

1. `degradation_document.py` + synthetic invoice + `doc_data.py` + `eval_document.py` smoke
2. `document_ocr.py` (RapidOCR + Tesseract adapters) + `doc_metrics.py`
3. `document_restore.py` + unit tests
4. Wire `SmartUpscaler.upscale(..., mode="document")` without breaking `photo`/`vector`
5. `eval_document.py` compares document mode vs raw vs photo-smart
6. Only then touch `app.py`

No new neural training in that PR series.

---

## Appendix A — Phase A/B measurements (2026-09-20, frozen set)

**Setup:** 6 synthetic invoices @300 DPI (2480×3508), levels mild/medium/heavy,
150-DPI preview set for cross-checks; RapidOCR 3.9.2 (PP-OCRv6 small, CPU) vs
Tesseract 5.5.3 (`--psm 6`, CPU). CER strict (case/punctuation counted);
`token_err` is case/punctuation-insensitive exact token match.

**RapidOCR (300 DPI set):**

| method | CER | token_err | coverage | false alarm | invented |
|---|---|---|---|---|---|
| clean | 0.0000 | 0.0000 | — | — | 0 |
| **raw** | **0.0937** | 0.0251 | 0.667 | 0.000 | 0 |
| restore (clahe stream) | 0.1424 | 0.0554 | 0.500 | 0.000 | 7 |
| restore (gray stream) | 0.1431 | 0.0841 | 0.500 | 0.000 | 7 |

**Tesseract (300 DPI set):**

| method | CER | token_err | coverage | false alarm | invented |
|---|---|---|---|---|---|
| clean | 0.0894 | 0.0000 | — | — | 0 |
| **restore (gray stream)** | **0.1749** | 0.1746 | 0.911 | 0.193 | 26 |
| restore (sauvola) | 0.1874 | 0.2566 | 0.874 | 0.169 | 41 |
| raw | 0.3844 | 0.3093 | 0.920 | 0.179 | 0 |
| sauvola (direct) | 0.3541 | 0.3270 | 0.965 | 0.113 | 68 |

**Decisions taken from the data:**

1. **RapidOCR is the default engine and eats raw input** — classical hardening
   (CLAHE/Sauvola/denoise) *hurts* it on every set tested.
2. **Tesseract, when used, eats the restored grayscale stream** — 55% relative
   CER improvement over its own raw baseline (0.384 → 0.175).
3. Classical restore stays in the pipeline for: display image, Tesseract stream,
   and evidence for flags (restored tokens flag *more* errors with fewer false
   alarms than raw in the earlier preview set).
4. An engine that is 2× better than the other on clean pages (RapidOCR 0.0 vs
   Tesseract 0.089) means "both adapters" is about **language coverage**, not
   English CER.
5. Latency: RapidOCR ≈ 4.3–4.7 s/page CPU at 300 DPI (inside the <8 s SLO,
   outside the 4 s stretch goal; DirectML EP or a det max-side cap is the fix).

**Open items this data creates:**

- Real-photo sanity set (SROIE/CORD public receipts) — synthetic pages do not
  contain perspective + real shadows the way phones do.
- Digit re-pass (recognition-only on 2× crops) still to be measured; the
  dual-stream gate already flags disagreements without picking a winner.
- Coverage/false-alarm trade-off is tunable via `conf_threshold`; freeze after
  the first real-photo run.

### Appendix A.1 — digit re-pass measurement (same frozen set) — *verdict invalidated*

Re-reading digit tokens with the recognition-only head on 2× crops of the **same
image** returned identical text for **13/13** tokens on a heavy page and produced
**0 conflicts** across the set, at **+0.4 s/token** CPU.

**INVALIDATED (2026-09-20):** this was measured through a RapidOCR 3.x state-leak
bug — a rec-only call permanently set `use_det=False` on the engine, so later
full-page OCR returned zero tokens (see commit `fix(ocr): rec-only crop re-read
silently disabled detection...` and `tests/test_document_ocr_state.py`). The
"13/13 identical" result cannot be trusted; it may have compared empty outputs.
The re-pass must be re-measured after the fix (queued; see Appendix D for the
corrected sweep that now actually exercises it).

Decision so far: `recheck_digits` stays available but **off by default**; the
**dual-stream** comparison (raw pixels vs restored pixels) is the primary digit
evidence. Re-measure the re-pass before promoting it.

**RE-MEASURED (P4b, 2026-09-20) — see Appendix I.** With the engine fixed, the
re-pass genuinely adds conflicts, and the pre-registered queue gate decides its
default: top-5 precision reached only **0.325** (conditional on pages with
errors) against the **≥0.5 bar** → `recheck_digits` stays **off by default**.
The corrected per-mode numbers live in Appendix I and
`out/doc_p4b_repass.json`.

---

## Appendix B — Phase C status (2026-09-20)

**Shipped**

- `document_export.py` — searchable PDF (reportlab, invisible text at token
  boxes), flag-colored overlay PNG (green/amber/red + alt reading), transcript,
  OCR JSON. PyMuPDF avoided (AGPL).
- `cli.py --mode document` — image and PDF input (page range via `--max-pages`),
  `--ocr rapidocr|tesseract`, `--lang`, `--deskew`, `--repass-digits`,
  `--skip-existing`, JSON report. Status line matches the UX copy:
  `n tokens · k low-conf · c digit conflicts · t seconds`.
- Dual-stream gate wired into the CLI: primary stream per `RECOMMENDED_STREAM`,
  auditor stream compared with `compare_digit_streams` (flags only, no picking).
- `restore_document(..., deskew=False)` geometry-stable mode so raw-OCR boxes
  align with the display embedded in the PDF.
- Perf: illumination background now estimated at 1/4 resolution —
  A4@300dpi restore 19 s → **1.6 s**; harness CER unchanged (0.0937).

**Measured (frozen set, RapidOCR raw):** CER 0.0937, coverage 0.667, false
alarms 0.000 → **4.14 s/page** on CPU (inside the <8 s SLO, above the 4 s
stretch goal).

**Phase C criterion re-interpretation (honest):** "document pipeline wins on
synthetic CER" is not achievable through preprocessing when the OCR engine is
already stronger than the degradation model. The product's measurable wins are:
searchable PDF + honest flags (2/3 of errors flagged, zero false alarms), and
the Tesseract path (+55% relative CER). This is recorded here rather than
hidden by reporting a prettier image.

**Next:** Phase D (Gradio default = document, `SmartUpscaler(mode="document")`,
cheap page classifier) and the public-receipt sanity set (SROIE/CORD) for
real-photo validation.

---

## Appendix C — Phase D status (2026-09-20)

**Shipped**

- `SmartUpscaler(mode="document")` + `page_likeness()/is_document()` classifier:
  synthetic invoices score **0.72–0.77**, photos (De1, PrakashJI, foo1) and
  noise score **0.0** — clean separation with no models, no OCR.
- `document_pipeline.py` — one implementation for CLI and app (restore →
  dual-stream OCR → flags → outputs); deskew=True keeps boxes aligned by
  running primary OCR on the rotated display.
- **Gradio studio is document-first**: "Document (recommended)" tab (image or
  PDF, OCR engine picker, deskew, overlay/PDF/TXT toggles, transcript with
  "Review these" list) with the photo studio moved to "Photo (Advanced)".
- **OCR performance**: RapidOCR is now DirectML-enabled automatically when
  `onnxruntime-directml` is present, with a one-time warm-up. Measured on the
  RTX 2050: CPU **9–15 s/page**, DirectML 25 s first call then **0.78 s/page**.
  Full pipeline (restore + both OCR streams + export): **6.6 s cold / 3.8 s
  warm** per A4@150dpi page — inside the 4 s stretch goal when warm.
- 35 tests green (documents, routing, exports, CLI, app, photo regression).

**Remaining for v1**

1. ~~P4 Devanagari (ne/hi) with vendored models~~ **DONE** (Appendix J): language
   routing + cls fix + fixture + gates; GUI language selector and
   `eval_document.py --build-devanagari/--lang` included, so the P4 numbers are
   reproducible from the harness. P4b **DONE** (Appendix I). Vendoring the model
   for air-gap stays on the P7 checklist.
2. Phase E mixed-page router (text + logo + signature + photo on one page).
3. P6 multi-page PDF; P7 license gate + packaging + tag v1.1.0.
4. ~~Real-data validation + frozen eval + CIs~~ **DONE** (Appendix K): A1-A4
   shipped; remaining gaps recorded there — Devanagari digit recognition,
   flag calibration on Devanagari, real phone photos (A5, needs a photographed
   field set), GT-verified modern Nepali line crops.

---

## Appendix D — P1 real-pixel gate (2026-09-20): SROIE + CORD, 30 pages each

**Setup.** Real photos streamed from the Hugging Face hub (internal eval only,
never redistributed): `jsdnrs/ICDAR2019-SROIE` (line-level `words`) and
`naver-clova-ix/cord-v2` (structured JSON → text leaves). Real pages are not
degraded (`clean == degraded`). `digCER` = CER over digit-bearing tokens
("money metric"); `BAG` = order-insensitive variant (annotation order is not
guaranteed to be reading order); hallucination counts only tokens no method and
no annotation produced (peer-reference rule, because real GT is incomplete).

### RapidOCR — the only viable real-photo backend

| set / method | CER | bagCER | digCER | digBAG |
|---|---|---|---|---|
| SROIE raw | 0.3635 | 0.4012 | 0.2143 | 0.2672 |
| SROIE restore_clahe | **0.3603** | 0.3941 | **0.2088** | 0.2436 |
| SROIE restore_gray | 0.3618 | **0.3921** | 0.2123 | **0.2408** |
| CORD raw | 0.5872 | 0.5357 | 0.2301 | 0.1689 |
| CORD restore_gray | **0.5748** | **0.5292** | 0.2588 | 0.1884 |
| CORD restore_clahe | 0.5927 | 0.5326 | **0.2250** | **0.1559** |

Tesseract on the same pages: SROIE raw **0.4446** vs restore_gray 0.4541; CORD
raw **0.8997** vs restore_gray **1.5740** (and 919–1851 invented tokens). Its
synthetic win (restore_gray CER 0.1318 vs raw 0.3844) does not transfer to real
photos: **demoted to clean-scan / synthetic fallback**, never the real-photo
default.

### Pre-registered gate vs measured

| gate | verdict |
|---|---|
| real CER ≤ raw + 5% | RapidOCR pass (both sets); Tesseract fails CORD (1.57 vs 0.90) |
| digit CER ≤ raw | RapidOCR clahe **pass** (0.2088/0.1559 bag), gray fails CORD digits |
| hallucination ≤ raw | raw = 0; restore adds 21–49 peer-unconfirmed tokens / 30 pages — **fail (strict)** |
| coverage ≥ 0.55 | **FAIL** — RapidOCR token coverage 0.014–0.032 raw, 0.024–0.082 pipeline; digit coverage 0.04–0.13 |
| FA ≤ 0.25 | RapidOCR mixed (0.11–0.47 digit FA); Tesseract **fail** (0.44–0.48) |

**Stop-the-line triggered** (this plan's own rule). Flag-policy sweep on the same
60 pages (decision units = RapidOCR digit tokens; 964 units / 73 wrong on SROIE,
319 / 39 on CORD), after the engine-state fix made the re-pass real:

| policy | SROIE cov / FA | CORD cov / FA |
|---|---|---|
| conf < 90 | 0.25 / 0.44 | 0.18 / 0.42 |
| conf < 95 | 0.32 / 0.74 | 0.23 / 0.61 |
| 2× re-read differs (1.5× or 2×) | 0.36 / 0.40 | 0.13 / 0.29 |
| cross-backend, bbox-contained | 0.73 / 0.79 | 0.23 / 0.87 |
| cross OR re-read OR conf<80 | 0.77 / 0.78 | 0.36 / 0.81 |

**No available signal combination reaches ≥0.55 coverage at ≤0.25 FA on real
photos.** The old bar is recorded here as failed — it is not silently moved.

### Decisions (shipped in the same PR series)

1. **RapidOCR is the real-photo path**; Tesseract is a clean-scan/synthetic
   fallback. `RECOMMENDED_STREAM` unchanged (rapidocr=raw, tesseract=gray).
2. **Honesty = ranked review queue**, not binary flags: `token_risk()` /
   `review_queue()` rank flagged tokens (digit_conflict 3.0, digit_uncertain
   1.5, low_conf 1.0, disagreeing re-read +2.0); exposed in `status_line`
   ("N to review"), the JSON `review` array, and the CLI top-3 printout.
3. **Revised P1 metrics (evidence-based, replaces the failed bar):**
   - queue usefulness: share of true digit errors present in the top-5 per page
     (to be measured on future changes);
   - digit signal quality reported per signal (coverage/FA pairs stay in the
     tables above);
   - no regression on CER/bagCER/digCER vs raw for the chosen streams.
4. **Re-pass default stays off** until re-measured (Appendix A.1 invalidation).
5. **Photo tier_c rejected** and gated behind `artifacts/tier_c/ACCEPTED`
   (see `gcp/RESULTS.md`); x4plus remains the shipped photo default.

### Corrected priorities after P1

Tesseract's collapse and the flag reality change the plan order only slightly:
P2 reading order and P3 orientation are still the next language-agnostic wins
(they directly cut CER/bagCER by fixing order, independent of flags); P4
Devanagari next; **P4b** re-measure the 2× re-pass with the engine fix and fold
it into the review queue; P5 mixed-page router after.

---

## Appendix E — P2 reading order shipped (2026-09-20)

**Design (conservative, evidence-driven).** `document_layout.sort_reading_order`
repairs *confident* two-column layouts and otherwise returns the engine order
byte-for-byte. The conservatism is a measured requirement, not taste: aggressive
row re-grouping cost **+8–19% CER** on SROIE receipts, because receipts are
single-column and their detection order is already row-major.

Confidence rules (each one added after a measured failure):
1. gutter = vertical union gap no token crosses, ≥ 2× median line height;
2. both sides need ≥ 6 tokens (receipt totals tables have clean gutters but
   3–5 rows — columnizing them is wrong);
3. the right side must be ragged (sd(x1) ≥ 0.5·sd(x0)) — right-aligned value
   columns of fixed width (`RM 33.92`) are detected by content instead;
4. the right side must not be number-dominated (≥ 50% of tokens ≥ 40% digits);
5. full-width rows are structural band separators only when *standalone*
   (sharing a row with other tokens made receipts churn: +8% CER);
6. if no confident split fires anywhere, the input order is returned unchanged;
7. column-major output preserves engine order *within* each column.

**Measured** (`out/doc_two_column.json`; 4 pages, RapidOCR, mild/medium):

| method | CER | WER | digCER |
|---|---|---|---|
| raw (detection order) | 0.5048 | 0.5817 | 0.4371 |
| pipeline (+ reading order) | **0.0421** | **0.1070** | **0.1028** |

WER gate (pre-registered −15%) exceeded at **−82%**. SROIE and CORD 15-page
probes: pipeline output is byte-identical to raw (identity rule holds on real
single-column pages). 11 layout tests; full suite 57 green.

**Escape hatch:** `run_document_pipeline(reading_order=False)`,
`cli.py --mode document --no-reading-order`; `meta["reading_order"]` records it.

**Known limitation (documented):** a full-width title that shares its row with
another token (e.g. a page number) is not used as a band separator; such pages
fall back to identity. Acceptable until a real case appears.

---

## Appendix F — P3 orientation shipped (2026-09-20)

> **CORRECTED the same day — see Appendix G.** The central claim below
> ("RapidOCR reads 180°/90° pages as garbage") was a measurement artifact:
> order-sensitive CER hid perfect line text behind reversed line order. The
> OSD-first policy and the "never flip 180" rule it justified were wrong and
> have been replaced by an evidence-first design with full real-corpus gates.

**Evidence gathered before shipping** (`document_orientation.py` docstring):

- RapidOCR reads 180°/90° pages as garbage (CER **0.80–0.85** vs 0.09–0.39
  upright) while token confidences stay ~99 — OCR confidence cannot detect
  rotation, and the line classifier does not fix page-level 180°.
- Tesseract OSD is exact on clean pages ≥1500 px but WRONG on small receipts
  even upscaled (3/6 upright receipts called 180° at conf up to 4.8); a later
  synthetic batch scored as low as 4.55 — confidence alone does not separate
  across content, and one large upright receipt still scored OSD 180° at 4.9.
- No GT-free text feature separates upright from 180° OCR (aggregate token
  stats were identical on synthetic pages; receipts varied by noise only).

**Shipped policy (safe by construction, no blind flips):**

1. EXIF orientation applied at load (`load_image_bgr`) — phone photos,
   including 180° via EXIF orientation 3.
2. OSD auto-rotation ONLY for 90°/270° on pages with long side ≥1500 px
   (receipts/small scans excluded by size; detector geometry guarantees the
   axis, OSD supplies direction), conf ≥3.0.
3. 180° is never auto-rotated from OSD: the opinion is recorded
   (`source="osd-180"`) and surfaced as `orientation_suspect`.
4. 90°/270° on small inputs get the geometry suspect marker (vertical text
   lines ≥60%), no auto action.
5. `meta["auto_rotate"]` and `meta["orientation_suspect"]` on every result;
   status line shows `· orientation?`; CLI `--rotate auto|off`.

**Measured**

| check | result |
|---|---|
| 90°/270° auto-rotation, synthetic pages | **8/8** (conf 3.98–6.29) |
| upright receipts rotated | **0/30** |
| upright receipts flagged 180-suspect | 1/30 (honest flag, no damage) |
| CER after auto-rotation vs upright | equal (0.00 on clean synthetic) |
| original "≥95% of rotated pages incl. 180°" gate | **FAILED on 180° by design** — replaced by refusal + suspect flag; recorded, not hidden |

**Limitation (documented):** 180° for EXIF-less images needs a dedicated
orientation classifier (4-way, rendered pages) — parked with this evidence.
65 tests green (8 orientation).

---

## Appendix G — P3b orientation, evidence-first (2026-09-20, same day)

**What was wrong in Appendix F.** Three linked errors, all from one artifact:

1. "180° reads as garbage" — false. The text is *perfect*, line for line
   (verified: all 23 tokens of the synthetic page, `INVOICE` → `Thank you for
   your business.`); only the line ORDER is reversed (detection walks the
   flipped image top-to-bottom). Order-sensitive CER read this as garbage.
2. "cls does not fix page-level 180°" — false. The line classifier works
   perfectly (`INVOICE` flipped → classed `180` at conf 0.9999999, re-OCR
   returns `INVOICE` vs `INVIOCCE` without it). RapidOCR *applies* it but does
   not expose the labels; the backend now re-runs it on our crops.
3. "90/270° is garbage" — false. Detected boxes are perspective-unrotated, so
   the text is largely readable; the failure was page-level direction only.

**Shipped design (commit 06380fa + 10a379e).**

```
pass1 OCR -> votes = (frac180, hi-conf-180, n) from line classifier + geometry
  frac180 >= 0.6 AND hi-conf >= 0.4          -> rotate 180, one re-run
  vertical token fraction >= 0.6             -> OCR the 90°-cw probe
      probe vertical < 0.6: probe votes pick 270 (action rule) vs 90
      probe still vertical                  -> OSD fallback, else suspect
  frac180 >= 0.5 but not actionable          -> vote-ambiguous suspect (no act)
  else                                       -> upright
```

**Why two vote statistics**: raw frac180 alone cannot separate — the flipped
minimum (0.73, sroie_00010) sits BELOW the upright maximum (0.79, cord_00005,
a blurry 4096px photo where the classifier votes high BOTH ways). Requiring
high-confidence (>=0.9) votes as well keeps every classifier-hard page out.

**Measured (SROIE 30 + CORD 30, both orientations; synthetic 4 angles × 4 seeds)**

| gate | result |
|---|---|
| synthetic: decision + CER < 0.05 + no suspect flags | **16/16** (post-fix CER 0.000) |
| SROIE upright: false rotations / false suspects | **0/30 / 0/30** |
| SROIE rotated (90cw/90ccw/180): decisions | **90/90** |
| SROIE rotated: CER <= upright + 0.05 | **90/90** |
| CORD upright: false rotations | **0/30** |
| CORD up+flip fully correct | 29/30 — the miss (cord_00011, 10 blurry tokens) is `vote-ambiguous` flagged, never flipped |
| passes per page | upright 1 (no extra OCR), 180: 2, 90ccw: 2, 90cw: 3 |
| votes overhead | ~0–300 ms/page (cls pass on ~50 crops) |

**Honest limits:** the ambiguous zone between the two clusters (frac 0.5–0.73
with low hi-conf) exists on blurry pages; those get the suspect flag instead
of an action. OSD remains the fallback for the tesseract backend (no line
classifier) and for inconclusive probes. 71 tests green (13 orientation).

---

## Appendix H — P2 real-layout verdict (2026-09-20)

**Why this appendix exists.** Appendix E's −82% WER was measured on a fixture
this repo renders; the sorter had never seen a real multi-column page. Fixed
by an internal-only eval set (never redistributed): 8 arXiv PDFs downloaded
once into `data/doc_eval/pdf_sources/` (7 confirmed 2-column CVPR/ICCV/ECCV +
Attention as a single-column control; layout verified per-paper via text-layer
line extents), rendered at 200 dpi with mild/medium degradation, first 4 pages
each → `data/doc_eval/real_two_column` (32 pages).

**What the real pages broke.** The union-gap gutter rule had no operating
point: OCR line boxes extend into the gutter (clean gaps 8–52 px at 200 dpi vs
a 2×-line-height = 78–80 px requirement) and a single wide token zeroed the
page-wide gap. Replaced with a **coverage gutter** (`document_layout.py`):
widest x-run crossed by ≤5% of boxes, ≥ max(8 px, 0.3 × median line height),
centred in the content; existing guards kept (≥6 tokens/side, right side
ragged, not number-dominated) plus each side must span ≥30% of content width —
a receipt label strip (24%) had columnized a SROIE page (`sroie_00003`);
sweeping 0.2–0.4 showed 0.3 keeps every true split and removes the false one.

**Measured (order-isolated, one OCR per page)**

| corpus | result |
|---|---|
| SROIE + CORD real single-column (60 pages) | **60/60 unchanged** (identity gate) |
| arXiv real 2-column (32 pages) | splits fired on 19/32 (13 are figure/table-heavy or the 1-column control, left as identity) |
| ... on those 19 pages | WER **0.870 → 0.268** (−69.1%), CER **0.702 → 0.086** (−87.7%) |
| regressions | **none** (no page worse by >2% WER) |
| shipped pipeline, whole set | CER 0.6065 → **0.2407**, bagCER 0.2524 unchanged (order repaired, content preserved) |

**Telemetry** (commit 19e9aaf): `meta["reading_order_splits"]` and
`meta["reading_order_changed"]` so future regressions are observable.
13 layout tests (2 new: thin-label-strip identity, box-padding tolerance).

**Honest limits:** the 13 unsorted pages include figure/table-heavy pages where
the band/wide-token rules bail out safely; the text-layer GT is content-stream
order (its floor noise is measured by `clean@rapidocr`, heavy on equation
pages). 3-column generalization remains deferred until a case demands it.

---

## Open truth items (recorded 2026-09-20, truth check)
Three gaps were found auditing the claims against the reports; none are hidden
in the appendices above:

*(all three resolved in Appendix I below; kept as the record of what was
unknown at the time)*

1. **Latency gate unmet on dense pages.** The shipped pipeline runs 8.0 s/page
   on the arXiv two-column set (dual-stream OCR + votes + restore) against the
   pre-registered <= 4 s/page warm gate. Measured cost lines: OCR ~1.1-2.2 s,
   votes ~0-0.3 s, restore + audit stream the rest. Deferred to P5/P7 with a
   known lever (skip the audit OCR pass on pages without digit tokens).
2. **rapidocr primary stream under-evidenced.** `RECOMMENDED_STREAM` says "raw"
   on evidence that only rules out clahe, while current reports show
   restore_gray better by CER on all three corpora (synthetic 0.0150 vs 0.0937;
   SROIE 0.3618 vs 0.3635; CORD 0.5748 vs 0.5872) but worse by bagCER on
   synthetics (0.1049 vs 0.0269). Re-measured in P4b; the table changes only if
   no corpus regresses on CER *and* bagCER.
3. **Review-queue usefulness never measured.** Appendix D set "share of true
   digit errors in the top-5 per page" as the revised P1 metric; no number
   existed yet. Measured in P4b (see Appendix I).

Also stale and fixed with this record: the `--repass-digits` help ("measured
redundant" - invalidated by the state-leak bug) and `--rotate` help (removed
OSD-first policy), commit f4d325b.

---

## Appendix I — P4b: re-pass, queue usefulness, stream re-verify (2026-09-20)

Resolves the three open truth items. All numbers are reproducible:
`out/doc_p4b_repass.json`, `out/doc_p4b_stream.json` (definition below).

**Setup.** RapidOCR on SROIE+CORD degraded pages (60). A digit-token *error* is
an OCR token whose digit sequence (all digit runs concatenated) is absent from
the page's GT digit multiset; 224 errors on 48/60 pages. The review queue is
`review_queue()` (risk ranking: flags × weights + re-read disagreement). The
pre-registered enable gate for the re-pass was: **top-5 precision >= 0.5** and
no CER/bagCER/digCER regression (text is never changed by the re-pass, so only
the first half can fail).

| mode | micro R@5 | micro R@10 | digit coverage | digit FA | P@5* | R@5* |
|---|---|---|---|---|---|---|
| off (shipped) | 0.049 | 0.049 | 0.049 | 0.353 | 0.208 | 0.052 |
| re-pass conf<95 | 0.121 | 0.121 | 0.121 | 0.386 | 0.312 | 0.132 |
| re-pass all | 0.228 | 0.268 | 0.268 | 0.434 | 0.325 | 0.216 |

\* mean over the 48 pages that have >=1 digit error (precision: of the top-5
items, the share that are true errors; recall: of the true errors, the share in
the top-5).

**Verdict 1 — `recheck_digits` stays off by default.** Best top-5 precision
0.325 < 0.5. The queue is honest but weak: it surfaces ~1 in 4 digit errors at
best (R@5 0.216) and the ranker needs better signals. Recorded, not hidden —
this is the revised P1 honesty metric finally measured, and it fails its bar
the same way the original coverage bar did.

**Verdict 2 — `RECOMMENDED_STREAM["rapidocr"]` stays "raw".** A/B via the new
`primary_stream` override under the shipped pipeline (reading order on):

| corpus | raw CER / bag / digBAG | restored CER / bag / digBAG |
|---|---|---|
| synthetic (6 p) | 0.0937 / **0.0269** / **0.0866** | **0.0917** / 0.0906 / 0.2023 |
| SROIE (30 p) | 0.3635 / 0.4012 / 0.2672 | **0.3614** / **0.3917** / **0.2402** |
| CORD (30 p) | 0.5872 / 0.5357 / **0.1689** | **0.5856** / **0.5304** / 0.1874 |

Restored wins CER marginally everywhere, but on clean synthetics it costs
bagCER 3.4x (0.0269 -> 0.0906) — token merging that CER hides. Gate was "no
corpus regresses on CER *and* bagCER" → **fails** → raw stays.

**Verdict 3 — the perf gap stands** (8.0 s/page dense 2-col vs <= 4 s gate,
recorded in the open items) — deferred to P5/P7 with the known lever (skip the
audit pass on digit-free pages).

**Cleanup in the same pass** (commit 2fc1b8a): 15 orphaned photo-era scripts
-> `legacy/` (self-contained root-path guards, README, smoke-tested),
3 root tests -> `tests/` (single 77-test run), `requirements.txt` split into
runtime + `requirements-dev.txt` (a fresh install previously could not run
document mode: rapidocr/reportlab/pypdfium2 were missing), README with the
measured capability table, 26 regenerable root PNGs deleted.


---

## Appendix J - P4 Devanagari (Nepali/Hindi), shipped 2026-09-20

**Goal.** `--lang ne|hi` with a vendored PP-OCRv5 Devanagari model, an exact-GT
synthetic fixture, and the pre-registered gate: synthetic CER < 0.15.

**Discovery chain (each step measured, not assumed).**
1. PIL cannot shape Devanagari (no Raqm): unshaped rendering made both the
   Latin and Devanagari models read gibberish. GDI+ rendering was worse.
   Qt/PySide6 (HarfBuzz, offscreen) shapes correctly - verified via
   QTextLayout glyph counts ('???' 3 codepoints -> 1 glyph).
2. RapidOCR's PP-OCRv6 has no Devanagari rec model: the engine needs
   `Rec.lang_type=devanagari`, `Rec.ocr_version=PP-OCRv5`,
   `Rec.model_type=mobile` (passed as Enums, not strings).
3. The 0/180 textline classifier mangles Devanagari crops - it flipped lines
   and page CER went 0.537 -> **0.039** with `use_cls=False` (isolated crops
   read perfectly either way, which is how the classifier was caught).
4. Page-level "recognition failures" on isolated words were a *detection*
   artifact; and the pipeline's reading-order sorter columnized a Hindi
   invoice table (fix in commit 724020a: sides must be filled by long lines).

**Shipped.** `normalize_lang` aliases (ne/nep/nepali, hi/hin/hindi), per-
language engine cache, cls disabled for Devanagari, Devanagari digits counted
by the honesty flags, `line_orientation_votes` unavailable for non-default
languages (sideways direction falls back to OSD), CLI `--lang`, fixture
renderer/builder, 6 new tests.

**Measured** (12 pages/script @300dpi, `out/doc_p4_devanagari.json`)

| set | ne | hi |
|---|---|---|
| clean, Devanagari engine | **0.030** | **0.028** |
| clean, Latin engine (control) | 0.859 | 0.833 |
| degraded mild | 0.028 | 0.059 |
| degraded medium | 0.084 | 0.100 |
| degraded heavy | 0.169 | 0.352 |
| degraded mean | **0.094 (PASS)** | 0.170 (partial) |

**Verdict: partial pass, recorded.** Nepali passes the <0.15 gate on all
levels; Hindi passes mild/medium and fails heavy (0.45-0.65x downscale
destroys small Devanagari matras). The restored stream does not help
(0.217 vs 0.202 ne at 200dpi). The Latin-engine control confirms the language
model is the reason it works at all.

**Limitations / next.** 180-degree Devanagari pages are not auto-corrected
(no cls votes; EXIF still covers phone photos). Hindi heavy needs either a
better detector for degraded text or a server-size Devanagari rec model
(none exists for PP-OCRv5 devanagari at server size). Vendoring the model into
`models/` for air-gap packaging is a P7 item (`Rec.model_path` /
`Rec.rec_keys_path` accept local files).

---

## Appendix K — real Devanagari evidence: frozen sets + CIs (2026-09-21)

> **Correction (N phase, same day).** The `nepali_pdf` row below is scored
> against a *corrupt* text layer. The N2 gate (`doc_metrics.devanagari_validity`)
> found 41/41 pages above the 2% invalid-sequence bar (page mean 22.4%,
> per-doc 8.5-40.1%; wrong ToUnicode maps: dha→i-matra, matra reordering,
> virama-for-space) and a ~60-document source hunt across ~25 government
> sites found no clean batch. **CER 0.338 is therefore not a
> recognition-error estimate** - it measures GT damage as much as OCR error.
>
> **Anchor (N5b, Gemini 2.5 Pro GT via Vertex AI).** All 41 v2 pages were
> transcribed by Gemini 2.5 Pro (2 strips/page); the model's Devanagari is
> clean (invalid-sequence rate 0.0002) and bodhan corroborates it on every
> page (bag agreement mean 0.881, min 0.750). Against this reference RapidOCR
> lands at **CER 0.135 [0.097-0.186] / bagCER 0.142 [0.111-0.186]**,
> valid-token recall **0.877 [0.732-0.968]**, per-doc bagCER 0.060-0.200.
> Provenance is explicit: this GT is *model-produced*, corroborated by a
> second model family, and the human pass is a disagreement review
> (`out/anchor_gemini/audit.html`), not a transcription. The earlier
> engine-vs-engine worksheet (`out/anchor/`) stays as the historical record.
> Digit claims are re-scoped in Appendix N. v1 numbers stay as the historical
> record.

The reality check's top two gaps were: (1) no real Nepali data anywhere in the
P4 claim, (2) all numbers were point estimates on small sets that the
thresholds had been tuned on. Both are addressed here.

### D1. Frozen-eval discipline (B)

`eval_freeze.py` pins every page/reference file by size + sha256 and records
license/provenance; `--frozen` makes scoring *refuse* when a file drifts.
Contract: thresholds are never tuned on a frozen set; any dataset change is a
new freeze version. `doc_metrics.bootstrap_ci` + `--bootstrap N` attach 95%
percentile CIs (deterministic per stable seed) to CER/bagCER/digCER/WER. The
harvesters/loaders gained `imwrite_safe`/`imread_safe` after discovering that
`cv2.imwrite` mojibakes non-ASCII paths (Devanagari-named pages landed on disk
under mangled names invisible to `os.path.exists` and `eval_freeze`; only cv2
could read them back). 103 tests green.

### D2. Datasets acquired (A1-A4)

| set | source | license | content | GT quality |
|---|---|---|---|---|
| `heidata_printed` | heiDATA doi:10.11588/data/EGOKEI | **CC BY 4.0** | 69 pages, 7 letterpress books (Hindi/Sanskrit/Braj), 1451 lines | human-corrected Transkribus ALTO (text + line boxes) |
| `nepali_pdf` | supremecourt.gov.np, lawcommission.gov.np | gov publications (internal eval) | 41 pages from 6 born-digital PDFs (statutes, judgments) | PDF text layer, Unicode-verified (>=0.4 Devanagari ratio gate; rejects scans and Preeti mojibake) |
| `nepali_lines` | HF `himalaya-ai/nepali-deva-ocr-eval` (from `gauravgiri/nepali-ocr-dataset`) | **unknown — unverified provenance** | 500 real Nepali print line crops | machine-generated, partly misaligned (probe evidence: 2640px crop with 18-char label). Provisional until a human pass |
| `nepali_unlabeled` | archive.org public items | public (no GT) | 19 pages (book, 2 newspapers, 1954 inventory) | none — behavior audit only |

### D3. Measured (devanagari engine, `raw` stream, frozen sets, 95% CIs)

| set | n | CER | bagCER | Latin-engine control | notes |
|---|---|---|---|---|---|
| rendered fixture (P4) | 12 | 0.030 | — | 0.86 | synthetic; now a *floor*, not a claim |
| `nepali_lines` | 500 | 0.717 [0.693-0.740] | — | 0.976 [0.968-0.984] | GT caveat above; exact match 1%; 66/500 empty |
| `nepali_pdf` | 41 | 0.338 [0.305-0.378] | 0.482 | — | per-doc 0.259-0.555; digit CER 0.289 |
| `heidata_printed` | 69 | 0.434 [0.387-0.481] | 0.553 | 0.96 | per-book 0.222 (jacobi1897) - 0.659 (pyarelala1914); digit CER 8.95 (Devanagari digits -> Latin insertions) |

Detection on `heidata_printed` vs ALTO boxes: **area coverage 0.975**
(center-extra 0.288; greedy IoU 0.459 is a line-vs-word granularity artifact).
Detection is not the bottleneck; recognition is.

Honesty signals on real Devanagari are broken: token-flag coverage 0.002
(`nepali_pdf`) / 0.016 (`heidata`) with ECE 0.82 — the GUI's confidence
colours carry almost no information on this domain. Devanagari digits must be
treated as unread (mobile rec model) until a digit-specific fix exists.

`nepali_unlabeled` behavior audit: 19/19 upright, 0 orientation suspects,
4/19 reading-order repairs — all four verified as genuine two-column splits
(46/37 and 54/55 tokens at 0.72/0.95 y-overlap); multi-column newspapers
stayed identity (>2 columns is a documented non-goal). Latency median 6.2
s/page on real scans (the known perf gap).

### D4. Verdict

P4's "<0.15 synthetic CER" pass does not transfer: on real Devanagari the
same engine lands at **CER 0.34-0.43**. The language switch is decisive
(Latin control 0.96-0.98), detection is fine, but the mobile recognizer,
Devanagari digits, and flag calibration are the product gaps. Nothing here is
hidden in a mean: per-document ranges, GT provenance, and the sets where no
CER may be quoted are all recorded. Still unproven: real **phone photos** of
Nepali documents (A5, needs a photographed field set) and GT-verified modern
Nepali line crops.

---

## Appendix L — Devanagari OCR bake-off: reuse before building (2026-09-21)

Goal: replace the weak mobile recognizer with an existing open-source model
instead of training anything. All candidates scored on frozen sets only
(`heidata_printed_v1`, first 150 digit-bearing lines + 69 pages), baseline
RapidOCR PP-OCRv5 devanagari mobile (digit-exact 0.533, line CER 0.316, page
CER 0.4335). Full table with CIs: `evals/bakeoff_results.md`.

| candidate | license | result | verdict |
|---|---|---|---|
| Tesseract 5 + tessdata_best nep | Apache-2.0 | lines CER 0.741 / digit-exact 0.207; pages CER 0.353 but **5,080 invented tokens** (532 digit victim) and 3x slower | **no** — invented gate fails hard; page CER alone is misleading |
| TrOCR-Devanagari-2 | MIT | lines CER 1.043 / digit-exact 0.020 | **no** — trained on handwritten words; hallucinates plausible unrelated Devanagari on print |
| GLM-OCR base (zai-org) | Apache-2.0/MIT | fits 4 GB (2.26 GB peak) but line mode out of distribution; pages CER 0.581/0.954/0.889, degenerate repetition loops, **196-229 s/page**; repetition-penalty tuning worse (0.700) and leaked Latin garbage | **no** — fails quality and speed; the base's Devanagari is weak (consistent with the unlicensed community fine-tune existing) |
| `himalaya-ai/glm-ocr-devanagari-finetuned` | none | not run | skipped: no root weights (6 x 3.57 GB checkpoints), unlicensed, shippable base already fails |
| bodhan-ai/indic-ocr | custom, gated | not run | blocked on `hf auth login` (user-owned token) |

Also measured as part of the same probes (recorded even though no candidate
won): digit-crop upscaling +2.7pp; logits-masked digit decode catastrophic
(0.553 -> 0.040, masked blank produced garbage runs); `script_mismatch`
signal (Latin token on a Devanagari page) fires on 20.4% of real tokens and is
~100% wrong/noise; conf<80 flags only 18.7% of in-script digit errors.

**Conclusion.** No existing open model replaces the recognizer on this
hardware/language pair today. The honest product path is not a new engine but
*flags and calibration*: make the review queue catch what the engine cannot
read (F-phase), keep the "never invent digits" promise, and record that
Devanagari digit reading remains the headline gap. The bake-off harness stays
in the repo (`eval_models.py --list`) so any future model can be scored on the
same frozen sets in minutes.

**Addendum (M4, bodhan accepted).** One candidate was still pending at first
publication of this appendix: bodhan-ai/indic-ocr (gated; the user accepted
the Indic Open Model License 1.0 — self-hosting incl. commercial is
permitted, third-party hosted access is not, attribution required). Frozen
results: digit-exact 0.653 on the 150 lines (baseline 0.533), conflict-flag
vs baseline digit errors recall 0.714 / precision 0.806, 0.44 s/line, 2.15 GB
VRAM — **the verifier gate passes**; page mode is 329-434 s/page, so the
primary role fails. Shipped as an opt-in verifier (commit 865a814): flagged
digit tokens get a second read; disagreement raises `cross_model_conflict`
with the alternative reading kept in `alt_text` (text never changed). Frozen
queue effect: R@5 0.533 -> 0.613, R@10 0.730 -> 0.752. The precision bar
(>=0.35) is still missed; recall improvements are real.

---

## Appendix M — Devanagari flags + calibration (2026-09-22)

With the bake-off showing no reusable model wins (Appendix L), the honest
product path is telling the user *where* the engine is wrong. Tuning happened
exclusively on a held-out dev set (`heidata_dev_v1`, 4 books / 61 pages,
never used for evaluation); all results below are frozen-set evaluations.

### Signals added

| signal | dev evidence | frozen evidence |
|---|---|---|
| `script_mismatch` (Latin token on a Devanagari page, weight 2.5) | 15.3% of tokens, ~100% junk | letterpress: 73 tokens, **precision 1.00**; gov PDFs: 46 tokens, precision **0.37** (Latin is legitimate there) |
| digit confidence bar 80 -> 90 (devanagari) | digit-error recall 0.235 -> 0.686 (precision 0.60 -> 0.31) | part of the queue numbers below |
| isotonic `cal_conf` (0-100) | dev ECE 0.640 -> **0.105** | letterpress ECE 0.818 -> 0.297; PDFs 0.821 -> 0.411 |
| `cross_model_conflict` (opt-in bodhan verifier, weight 3.0; Appendix L) | line-crop flag recall 0.714 / precision 0.806 | queue R@5 0.533 -> **0.613**, R@10 0.730 -> **0.752** (32 conflicts / 69 pages on flagged suspects) |

Calibration note: temperature scaling was measured to be *structurally
unsuitable* on this domain — probabilities saturate near 1 while token
accuracy is 0.28, so the BCE optimum inverts the ranking (T=-7.75) instead of
fixing the scale. The shipped map is isotonic (monotone, so ranking/AUC 0.733
is untouched); `fit_calibration.py` records the dev manifest sha256 and the
temperature evidence in the JSON.

### Frozen review-queue quality (`eval_flags.py`)

| set | digit-token err rate | R@5 | R@10 | P@5 | P@10 |
|---|---|---|---|---|---|
| heiDATA letterpress (69 p) | 0.333 | **0.533** | **0.730** | 0.212 | 0.160 |
| gov PDFs (41 p) | 0.316 | 0.154 | 0.154 | 0.261 | 0.240 |

Pre-registered bar was R@10 >= 0.4 at P >= 0.35: **recall passes on the hard
scan set (0.73) and precision misses**; **on clean PDFs both miss** — their
errors are confidently wrong (ECE 0.82), so confidence-based signals barely
fire. The queue is a large honest improvement for degraded scans (P4b's
receipt baseline was R@5 0.049) and weak for born-digital pages; the
difference is now a measured product fact instead of an assumption.

Ranker comparison on dev (current risk+reading-order vs confidence tiebreak vs
continuous digit-risk): current wins (R@5 0.537 vs 0.423 / 0.504), so the
existing ranking stays.

### What this changes for the product

- Review queue on Devanagari scans now surfaces ~3 of 4 digit errors in the
  top 10 (was ~1 in 20 on the receipt baseline); the JSON `review` list, CLI
  top-3 and GUI overlay all inherit it.
- Calibrated confidence is reported (`cal_conf`, meta `calibration`) and used
  for honest ECE reporting; it does not silently move text or thresholds.
- Devanagari digit *reading* remains unsolved (Appendix L); the product
  position is "flagged, not invented".

---

## Appendix N — recognition & digit reality; fine-tune recipe (2026-09-21, R phase)

### N1. What fails: error taxonomy on human GT (R1)

`eval_error_taxonomy.py` classifies token errors between RapidOCR and the
human-corrected ALTO GT (heiDATA, 69 pages / 7 books, frozen) into digit /
matra / consonant / order / segmentation / missing / invented / other.

| book | lines | gt tok | err rate | seg | miss | cons | matra | digit | det_x |
|---|---|---|---|---|---|---|---|---|---|
| diksita1895 | 51 | 232 | 0.388 | 26 | 32 | 9 | 8 | 0 | 34 |
| jacobi1897 | 212 | 1822 | 0.336 | 46 | 377 | 119 | 30 | 2 | 110 |
| jagannatha1955 | 210 | 1360 | 0.265 | 141 | 49 | 80 | 37 | 9 | 101 |
| jayadeva1926 | 255 | 1271 | 0.344 | 166 | 124 | 65 | 36 | 5 | 135 |
| pyarelala1914 | 206 | 1243 | 0.439 | 152 | 181 | 42 | 24 | 3 | 114 |
| sankaracarya1925 | 238 | 1885 | 0.222 | 128 | 20 | 136 | 74 | 2 | 71 |
| sivaramasukla1900 | 279 | 1848 | 0.329 | 219 | 99 | 102 | 97 | 0 | 91 |

Segmentation + missing dominate (line assembly and detection, not just
recognition); consonant and matra errors are next; digit errors are rare in
letterpress text. `det_x` = OCR tokens whose center falls in no ALTO line
(headers/footers/noise). Order errors: 0.

### N2. Digit reality, scoped by domain (R2)

| domain | set | valid-token recall | digit CER | digit-exact pages | queue R@10 |
|---|---|---|---|---|---|
| modern gov PDFs (**anchor**, Gemini GT) | `nepali_pdf_v2` (41 p) | 0.877 [0.732-0.968] | 0.126 | 0.073 | 0.154 |
| modern gov PDFs (corrupt text-layer GT, historical) | `nepali_pdf_v2` (41 p) | 0.522 [0.480-0.562] | 0.289 [0.227-0.352] | 0.049 | 0.154 |
| letterpress scans | `heidata_printed` (69 p) | 0.647 [0.619-0.675] | 8.95 [5.86-12.50] | 0.000 | 0.730 |

Digits are broken in *different* ways per domain: letterpress digits are
essentially unread (the model emits Latin insertions - digit CER 8.95), while
on clean PDFs most digits are individually readable (digit CER 0.126) but
whole-page digit sequences are exactly right on only 3/41 pages. The review
queue helps on scans (R@10 0.73) and not on clean PDFs (0.154) - their errors
are confidently wrong.

### N3. Verifier coverage, pre-registered (R3)

`--digit-verifier` gained a scope: `flagged` (default) or `all` digit tokens.
Adopt-if was pre-registered: queue R@10 **+>=3pp** on frozen heiDATA at
**<=+3 s/page**.

| scope | conflicts | R@5 | R@10 | s/page | verdict |
|---|---|---|---|---|---|
| off (baseline) | 0 | 0.533 | 0.730 | 2.24 | - |
| flagged (default) | 32 | 0.613 | 0.752 | 10.80 | shipped opt-in |
| all | 59 | 0.635 | 0.788 | 18.20 | recall +3.7pp PASS, cost +7.4 s/page FAIL -> stays opt-in |

Note the cost table: per-page batch overhead dominates the 0.44 s/token
estimate (model load + one batch call per page). The default stays `flagged`.

### N4. Fine-tune recipe (documented, not started)

Target the taxonomy, not the mean: segmentation/missing first, then
consonants/matras, digits last (but gated hardest).

- **Data**: (a) synthetic Devanagari lines from our Qt renderer
  (`doc_data.render_devanagari_invoice` + a line-crop exporter) with exact GT
  and controllable degradation; (b) heiDATA ALTO line crops (human GT, CC BY
  4.0); (c) line crops from the v2 PDFs, **GT only where the text layer passes
  `devanagari_validity`** (corrupt tokens are never training targets).
- **Model**: fine-tune a small recognizer head (PP-OCRv5 mobile rec or TrOCR
  small) on the frozen dev split `heidata_dev_v1`; tuning only on dev, never
  on frozen eval sets.
- **Gate (pre-registered)**: adopt only if, on frozen `heidata_printed_v1` +
  `nepali_pdf_v2`: digit-exact **>= 0.75** (baseline 0.0 letterpress / 0.049
  PDFs) and bagCER not worse than baseline (letterpress 0.553 / v2 valid-subset
  recall not worse), at <= +1 s/page.
- **Not started**: no training run is scheduled by this appendix; it exists so
  the next attempt starts from a pre-registered bar instead of an aspiration.

---

## Appendix O — multi-page combined PDF (P6, 2026-09-21)

`write_searchable_pdf_pages` writes one PDF page per `(image, tokens, dpi)`
entry, each keeping its own page size. CLI: `--max-pages 0` = all pages, and
the run additionally writes `<stem>_combined.pdf` + `<stem>_combined.txt`
(per-page artifacts unchanged; default stays `--max-pages 1`). GUI: an
"All pages (PDF)" checkbox (default off; slider/overlay show page 1, the
combined files cover all). Verified by tests: a 3-page demo PDF yields a
3-page combined PDF with extractable text on every page and correct sizes;
full suite 189 green - no single-page regression.

---

## Appendix P — mixed-page router gates (P5, 2026-09-21)

### Setup

`mixed_pages` (synthetic, `doc_data.render_mixed_document`): 6 pages @300dpi
with text + logo + photo + signature, exact text GT and region sidecars; photo
textures are DIV2K crops when present, procedural otherwise. Text regions for
the router come from OCR boxes; everything else is composited from the
original. Recognition is unchanged (OCR still runs on the raw page).

### Pre-registered gates and measured result

| gate | bar | measured | verdict |
|---|---|---|---|
| text CER | <= plain * 1.02 | 0.2664 -> 0.2651 (no regression) | PASS |
| non-text PSNR/SSIM | >= naive full-page restore | 60-67 dB vs 18-23 dB (router keeps originals) | PASS |
| invented tokens | 0 | 1 word (`barkrddeo.com`, detector false-positive over untouched photo texture) | **FAIL** |
| cost | <= +1 s/page | 0.31 s/page | PASS |

**Verdict: not all gates passed -> the router stays opt-in**
(`--mixed-router`), it is not auto-enabled. The value it protects is large
where it applies (non-text PSNR 60-67 dB vs 18-23 dB for a naive full-page
restore), and the failure is a single detection instability, not corrupted
pixels.

### Geometry fix found by this phase

The pipeline used to OCR >2500 px inputs at full resolution while the display
was fitted to `MAX_SIDE` - boxes (and PDF text) fell outside the exported
page. `run_document_pipeline` now fits the page once at entry, so display,
boxes and exported page share one coordinate space; `meta["resized"]` is now
real and the GUI's "downscaled" warning fires. Regression test:
`test_pipeline_boxes_stay_inside_downscaled_display`.
