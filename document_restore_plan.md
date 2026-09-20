# VectorScaling — Faithful Document Restore

**Status:** Product plan. Execute in order. Do not skip the eval harness.
**Decision date:** 2026-09-20
**Relationship to old roadmap:** This *replaces* photo-SR as the default product. The math roadmap (`math_based_image_upscaling_roadmap.md`) remains a research appendix. Phase 4.5 (vector/raster split) and Phase 8 (reconstruction constraint) are reused, aimed at pages instead of portraits.

---

## 1. Product one-liner

Local restore for mixed documents where a wrong character costs money.
Sharp when we are sure. Honest when we are not.

**Job to be done:** Drop a scan, phone photo of a bill, screenshot, or PDF page. Get a cleaner image **and** searchable text, with low-confidence glyphs marked instead of invented.

**Not the job:** 4× cinematic photos, anime, license plates, “looks sharper than Upscayl.”

---

## 2. Why this product (novelty × usability)

| Bet | Usable? | Novel? | Verdict |
|---|---|---|---|
| Photo 4× SR | Yes (Upscayl exists) | No | Kill as homepage |
| Anime SVG | Weak | Overclaimed | Kill |
| CCTV plates | Only with a dedicated camera | No | Kill |
| Generic “document upscaler” | Yes | No (Adobe Scan, Lens, DocRes) | Insufficient |
| **Faithful mixed-page restore + OCR confidence** | Yes (bills, contracts, archives, UI) | Yes as a *consumer local* product | **This** |

Novelty is not a new backbone. It is three things composed:

1. **Do not invent glyphs.** Reconstruction / agreement gate. Ambiguous `8`/`B` stays soft and flagged.
2. **Route the page.** Text → restore + OCR. Logos/stamps → vectors. Signatures/photos → freeze (no GAN).
3. **Dual output.** Searchable PDF (the work) + optional SVG of true graphics (the bonus).

Usability is one loop: drop file → preview with warnings → download PDF/text. Local by default (privacy is the reason to use us instead of Lens).

---

## 3. Non-goals (explicit)

- Do not train another Real-ESRGAN clone on DIV2K.
- Do not ship UltraSharp / Remacri / grain as document defaults.
- Do not vectorize letters as the primary representation (traced `A` blobs are not searchable).
- Do not claim forensic plate recovery or “enhance CCTV.”
- Do not require cloud APIs for v1.
- Do not optimize PSNR as the north-star metric.

Photo engines (`sr_engine.py`, ncnn models, `train_v2.py`) may stay in the repo under **Advanced / Photo**. They are not the product.

---

## 4. Users and slices

**Primary (v1–v2)**

- People digitizing bills, receipts, invoices (amounts must not hallucinate).
- Students / researchers with phone photos of papers and old PDFs.
- Anyone who needs a searchable PDF without uploading to a cloud.

**Secondary (v2+)**

- Mixed pages: letterhead + stamp + table + a small photo.
- UI / error-message screenshots (high-contrast type).

**Not a user yet**

- Industrial ALPR.
- Full archive shops that already run ScanTailor + ABBYY (we can learn from them; we will not beat them on batch in v1).

---

## 5. Success metrics

North star: **character error rate (CER)** on a frozen eval set, plus **hallucination rate**.

| Metric | Definition | v1 bar | v2 bar |
|---|---|---|---|
| CER | `edit_distance(ocr, gt) / len(gt)` | Beat raw Tesseract on the same page by ≥20% relative | Beat ScanTailor-ish baseline; competitive with DocRes *on our set* |
| WER | word-level edit rate | Track; secondary | Track |
| Hallucination rate | OCR tokens that do not appear in GT *and* were not in raw OCR either (invented digits/letters) | **Must not rise** vs raw OCR | Must fall |
| Ambiguity coverage | fraction of true errors that were flagged low-confidence | ≥50% of remaining errors flagged | ≥70% |
| Time | page-to-PDF on CPU, 1× A4 scan ~150–200 DPI | < 8 s median | < 4 s or GPU optional |
| Privacy | no network on the happy path | Required | Required |

PSNR/SSIM on glyphs may be logged. They must not decide shipping.

**Kill criteria:** if after Phase 3 (app wired) CER does not beat raw OCR + a simple adaptive-threshold baseline on *real* bills/scans we collect, stop adding models. Fix restore and OCR config first.

---

## 6. Eval harness first (do not skip)

Mirror the old Phase 0, with text as ground truth.

### 6.1 Datasets

**Synthetic (always on in CI)**

- Render 40–80 pages from public-domain / generated invoices (known UTF-8 GT).
- Fonts: Arial, Times, Courier, one Devanagari if we care about Nepali/Hindi later (Phase 4).
- Degrade with a *document* pipeline (new module, not `degradation_v2` photo kernels):
  - JPEG q=30–70
  - slight perspective / barrel
  - uneven illumination (smooth gradient + shadow)
  - Gaussian/motion blur 1–2.5 px
  - downscale 0.4–0.7 then upscale (phone photo)
  - optional speckle / paper texture *low amplitude*

**Real (offline, not in git if personal)**

- 20+ real pages: bills, book photos, screenshot UI, one stamped letter.
- Transcribe GT by hand into `data/doc_eval/gt/*.txt`.
- Never commit private bills. Use `data/doc_eval/` gitignored except synthetic.

### 6.2 Scripts

| New file | Role |
|---|---|
| `degradation_document.py` | Deterministic document degradations (seeded) |
| `eval_document.py` | Restore → OCR → CER/WER/hallucination vs GT |
| `data/doc_eval/synthetic/` | Rendered clean pages + GT |
| `tests/test_document_restore.py` | Unit tests on synthetic “INVOICE 1200.00” |

Baselines in the harness, always:

1. Raw image + Tesseract
2. Adaptive threshold (Sauvola) + Tesseract
3. Lanczos 2× + Tesseract
4. Current `SmartUpscaler(mode=auto)` + Tesseract (prove photo SR *hurts* or does nothing)
5. Our document pipeline (the thing we ship)

Ship a JSON report like `out/smoke_metrics.json`.

---

## 7. Target architecture

```
PDF / image
    │
    ▼
ingest (render PDF page @ 150–300 DPI, EXIF rotate)
    │
    ▼
page router (v2; v1 = treat whole page as text)
    ├─ text / table regions     → restore → OCR
    ├─ logo / stamp / badge     → existing Bézier vector (no OCR)
    ├─ signature / photo / stamp ink texture → freeze (identity or mild denoise)
    └─ background paper         → flatten illumination only
    │
    ▼
confidence map (per-glyph OCR conf + optional recon disagreement)
    │
    ▼
outputs
    ├─ restored PNG
    ├─ searchable PDF (text layer + image)
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
| `app.py` Gradio slider | Keep. Default mode `document`. Photo models → Advanced accordion. Grain default 0 (already). Hide UltraSharp as default. |
| `smart_upscaler.py` | Add `mode="document"` (and treat `auto` as document if page-like). Photo path remains `mode="photo"`. |
| `vector_raster_hybrid.py` | Keep. v2: run on *non-text* masks only (logos/stamps). Do not vectorize glyph interiors in v1. |
| `sr_engine.py` / x4plus / x4v3 | Advanced photo only. Not called from document default. |
| `deep_unfolding.py` fidelity | Candidate *inner* prior later if CER plateaus; not v1. |
| `eval_harness_v2.py` | Keep for photo regression; not the product scoreboard. |
| `train_v2.py` / DIV2K | Frozen unless Phase 5 opens. |
| `cli.py` | Add `--mode document`, PDF in/out, `--ocr`, `--lang`. |
| `test_smart_upscaler.py` | Keep photo/vector tests. Add document tests; do not break PrakashJI skin gates for `mode=photo`. |
| Examples De1 / PrakashJI | Move to “Photo examples”. Document examples: synthetic invoice + one public scan. |

---

## 9. Technical choices (defaults)

| Concern | v1 default | Why | Revisit |
|---|---|---|---|
| OCR engine | Tesseract via `pytesseract` | Local, boring, good enough to measure restore | PaddleOCR if Latin CER stalls or we need boxes/conf more easily |
| PDF read | `pypdfium2` | Lightweight render | PyMuPDF if write+read in one lib is simpler |
| Searchable PDF write | PyMuPDF (`fitz`) insert image + invisible text | One dependency for write | reportlab if fitz fights us |
| Restore | Classical OpenCV: rotate, illumination (background subtract / morphological black-hat), denoise (bilateral or NLM light), Sauvola binarize *as a side stream* (keep grayscale for display) | Matches ScanTailor’s actual gains; no training | DocRes weights only if harness says we lose on dewarp/shadow |
| Languages | `eng` + `osd` | Tesseract lang packs documented | `nep`/`hin` as v2 flag |
| Scale | Restore at native / 2× max | 4× is rarely the bottleneck | User toggle |
| UI | Existing Gradio | Already shipped | — |

**Display vs OCR:** show a cleaned *grayscale* page (readable). Feed OCR a *binarized or contrast-normalized* sibling. Do not force the user to look at a harsh binary unless they opt in.

---

## 10. Phases

### Phase A — Harness (week 1)

**Build**

- `degradation_document.py`
- Synthetic invoice renderer (PIL/cv2: title, table, total `1200.00`, address)
- `eval_document.py` with baselines 1–4
- `tests/test_document_eval_smoke.py` (2 pages, CER computed)

**Done when**

- `python eval_document.py --json out/doc_smoke.json` runs on CPU in CI-like time
- JSON includes CER for raw / sauvola / lanczos / smart-photo
- We have a written note: photo SR CER vs raw (expected: photo SR ≥ raw, i.e. not better)

### Phase B — Classical restore core (week 1–2)

**New module:** `document_restore.py`

API:

```python
restore_document(img_bgr, *, scale=1) -> dict
# keys: display_bgr, ocr_bgr, debug (illum, binary, skew_angle)
```

Steps (each function independently testable):

1. EXIF/array orientation; optional 90/180 via Tesseract OSD later
2. Downscale huge scans to max side 2500 for speed (record scale)
3. Estimate skew (min-area rect of text-like edges or projection profile); rotate
4. Illumination flatten (large-kernel median / morphological closing)
5. Light denoise (edge-preserving)
6. Build `ocr_bgr`: Sauvola or Wolf binary, or CLAHE+binary
7. Build `display_bgr`: illumination-corrected grayscale, *not* over-binarized
8. If `scale==2`, Lanczos on display; OCR on the 2× binary

**Done when**

- CER on synthetic degraded invoices beats raw and beats Lanczos-only
- Hallucination rate does not increase
- Unit tests: known skew 5°, known shadow gradient

**Do not** call `sr_engine` here.

### Phase C — OCR + confidence + files (week 2)

**New module:** `document_ocr.py`

- `ocr_page(ocr_bgr, lang="eng") -> list[Token]` with `text, conf, bbox`
- Flag `conf < 60` (tune on harness)
- Digit-run detector: tokens matching `[0-9]{2,}` get a second pass (Tesseract `--psm 7` on crop) ; disagreement → `flag="digit_conflict"`

**New module:** `document_export.py`

- Write `searchable.pdf` (image + hidden text at bboxes)
- Write `overlay.png` (boxes: green high conf, amber low, red conflict)
- Write `transcript.txt` and `ocr.json`

**CLI**

```
python cli.py --mode document --input scan.jpg --output out_dir --ocr --pdf
python cli.py --mode document --input pack.pdf --output out_dir --ocr --pdf
```

**Done when**

- PDF text is selectable and roughly aligned
- Overlay exists
- `eval_document.py` includes our pipeline as baseline 5 and wins on synthetic CER

### Phase D — App default (week 2–3)

**`app.py`**

- Default routing: **Document (recommended)**
- Inputs: image *and* PDF (first page in v1; “all pages” v1.1 if cheap)
- Outputs: slider (original vs display restore), overlay, transcript markdown, files (PNG, PDF, TXT). SVG checkbox: **off by default**; “graphics only”
- Remove grain slider from the document pane
- Photo engine dropdown lives under Advanced
- Examples: synthetic invoice + public document scan; portraits demoted

**`smart_upscaler.py`**

- `mode="document"` calls `restore_document` (scale 1 or 2)
- `mode="auto"`: cheap page classifier (see below); if document-like → document, else current photo+vector
- `mode="photo"` / `"vector"` / `"fidelity"` unchanged

**Page classifier (cheap, v1)**

- Gray, high connected-component count of text-sized blobs, low color saturation, little skin → document
- Else photo
- Bias: **prefer document** if uncertain (product is documents now)

**Done when**

- `test_web_app.py` still passes for a synthetic text image
- New test: invoice → overlay + pdf path returned
- Manual 10-minute pass on 5 real pages (not committed)

### Phase E — Mixed-page router (the wedge) (week 3–4)

This is the demo that is not “ScanTailor in a Gradio skin.”

1. Segment:
   - Text lines (MSER / EAST-lite / Tesseract layout / contour aspect)
   - Graphic blobs (existing `segment_flat_and_graphic_regions` minus text boxes)
   - Photo-like (high local variance, skin, or large texture) **and signatures** (ink, connected, in a typical signature band)
2. Text boxes → restore+OCR
3. Graphic blobs → current vector export (stamps, logos). Do **not** OCR them.
4. Signature/photo boxes → copy pixels through (maybe mild denoise). **No SR.**
5. Recompose display image
6. Digit-conflict gate on amounts (regex for currency / totals)

**Done when**

- A constructed page (photo thumbnail + invoice text + colored logo) OCRs the table, vectorizes the logo, leaves the photo un-GANed
- Test in `test_document_router.py`
- Hallucination rate on synthetic amounts still ≤ raw

### Phase F — Only if CER plateaus (optional research)

- Try DocRes (or similar) **weights as a module** behind the same API; keep the gate
- Small text-SR (TSRN-class) **inside text boxes only**, reconstruction-constrained
- Unfolding/fidelity as a digit prior — only if it *lowers* hallucination, not just PSNR

If a borrowed model wins CER without raising hallucination, **use it**. Do not retrain a photo GAN to “be DocRes.”

### Phase G — Product polish

- Multi-page PDF
- Language pack UI (`eng`, `nep`, `hin`, `fra`…)
- Batch folder in CLI (already almost there)
- “I’m not sure” summary at the top: list of flagged amounts/dates
- Desktop README: Tesseract install on Windows

---

## 11. UX copy (ship this tone)

- Title: **VectorScaling — Document Restore**
- Subtitle: Local. Searchable. Won’t invent the numbers on your bill.
- Primary button: **Restore & read**
- Status line: `CER proxy: n low-confidence glyphs · k digit conflicts · t seconds`
- If digit conflict: show original crop and restored crop side by side, neither auto-picked.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tesseract install hell on Windows | Document `winget install Tesseract-OCR`; skip OCR tests if binary missing; restore still works |
| Classical restore fails on heavy curl/dewarp | Phase F DocRes dewarp only; don’t block v1 |
| Users still judge “prettiness” | Overlay + transcript make OCR the visible product |
| Photo users feel abandoned | Advanced → Photo mode, old engines intact |
| Private eval data | gitignore; synthetic CI only |
| Scope creep (plates, anime) | This document’s non-goals; refuse in UI |

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

## 14. First implementation slice (when coding starts)

Do this in one PR, in order:

1. `degradation_document.py` + synthetic invoice + `eval_document.py` smoke
2. `document_restore.py` + unit tests
3. Wire `SmartUpscaler.upscale(..., mode="document")` without breaking `photo`/`vector`
4. `eval_document.py` compares document mode vs raw vs photo-smart
5. Only then touch `app.py`

No new neural training in that PR.

---

## 15. Open questions (locked defaults)

| Question | Default unless we revisit |
|---|---|
| Tesseract vs PaddleOCR | Tesseract v1 |
| Binarize for display? | No; grayscale display, binary for OCR |
| Default scale | 1× restore; 2× optional |
| `auto` means | Document if page-like, else photo |
| SVG default | Off |
| Languages | English first |
| Cloud | Never in v1 |

---

## 16. Definition of “we shipped the product”

A stranger can:

1. Install Tesseract + `pip install -r requirements.txt`
2. Open the Gradio app
3. Drop a phone photo of a printed invoice
4. Download a searchable PDF
5. See any shaky totals highlighted instead of silently “fixed”

And on the synthetic harness, document mode has **lower CER than raw and lower CER than current photo SmartUpscaler, with hallucination rate no worse than raw.**
