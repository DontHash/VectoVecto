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

1. P4 Devanagari (ne/hi) with vendored models; P4b re-measure the 2× re-pass
   with the engine fix and fold it into the review queue.
2. Phase E mixed-page router (text + logo + signature + photo on one page).
3. P6 multi-page PDF; P7 license gate + packaging + tag v1.1.0.

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
