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

### Appendix A.1 — digit re-pass measurement (same frozen set)

Re-reading digit tokens with the recognition-only head on 2× crops of the **same
image** returned identical text for **13/13** tokens on a heavy page and produced
**0 conflicts** across the set, at **+0.4 s/token** CPU. Identical input to the
same recognition model is redundant by construction.

Decision: `recheck_digits` stays available but **off by default**; the
**dual-stream** comparison (raw pixels vs restored-grayscale pixels) is the
mechanism that yields independent digit evidence. Candidates for a future
re-pass that would add signal: a *different* rec model, or a different image
(2× Lanczos of the raw page, not a crop of it).

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
