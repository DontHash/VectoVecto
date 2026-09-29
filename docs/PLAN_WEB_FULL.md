# Web app: full-feature plan + repo diet

**Status:** executing — the removal half landed (see the ledger below).
Basis: a read-only audit of the working tree on 2026-09-30 — product version
1.3.0 (`pyproject.toml:7`), HEAD `29b398d`, plus the (still untracked)
`webapp/`.

**Goal.** The web app (FastAPI `webapp/vvweb/` + SolidJS `webapp/frontend/`)
is the single UI. It reaches feature parity with the CLI's single-page
workflows — the document tab completely, the photo studio behind an explicit
opt-in — with one shared pipeline, no duplicated logic, and honest resource
caps. The repo stays cloneable and locally runnable, and nothing that is
unused stays in the product path.

## Status ledger

| Workstream | Status | Notes |
|---|---|---|
| Gradio removal (user decision) | **done** 2026-09-30 | `app.py` + its tests deleted; gradio dep, `vectovecto-studio` script, Docker, CI, packaging and docs updated |
| Markdown export (user request) | **near-finished** 2026-09-30 | `<stem>.md` in export/CLI/web, tests green; web smoke on a PDF passed (23 tokens → `transcript.md`) |
| P0 Repo truth & hygiene | **done** 2026-09-30 | frontend CI job, first `webapp` tests, Docker web image, docs pass; `webapp/` committed and pushed (`04d4eb2`) |
| P1 Diet & broken-path fixes | **done** 2026-09-30 | legacy + kaggle/vlm_eval removed (w1_train kept); fidelity removed; rrdbnet/deva_reader shipped; dead ONNX deleted; `--workers` removed; deps split (scipy stays — skimage). Held: CLI grain unify (P3) |
| P2 Web document parity | not started | engine choice, output toggles, all-pages, meta |
| P3 Web photo studio | not started | the remaining feature gap (CLI-only today) |
| P4 Efficiency & deploy | partially done | Docker now builds the web app; dependency split pending (P1) |
| P5 Verification | not started | |

**Feature maturity labels** used in docs and later in the UI: `stable` ·
`beta` (works, still changing) · `experimental` (opt-in, may change) ·
`planned` · `not shipped`.

| Feature | Label |
|---|---|
| Document pipeline (en/ne/hi), review queue, multi-page PDF | stable |
| Web document studio, Markdown export | beta |
| Photo upscale (CLI) | beta |
| Web photo studio | planned |
| Mixed-page router, Devanagari line reader, digit verifier (bodhan) | experimental |
| Fidelity (deep-SR) mode | not shipped (checkpoint never published) |

---

## 0. Answers first

### 0.1 What is unused, dead, or heavier than it should be

Sizes measured on this machine 2026-09-30.

| # | Finding | Evidence | Verdict |
|---|---|---|---|
| 1 | **`webapp/` is untracked** — the entire new UI is unversioned; GitHub `main` has no `webapp` path at all | `git status --short` → `?? webapp/`; `.gitignore:122-126` only anticipates its subpaths | **Commit it** (P0) |
| 2 | `legacy/` (pre-pivot research, cloud VM scripts) — zero importers | grep: no `import legacy` anywhere; only its own files import root modules | **Done 2026-09-30**: removed from the tree; git history keeps it |
| 3 | `kaggle/` VLM eval staging (the VLM mode was dropped) — nothing imports it | grep: no importers; 43 MB local, mostly ignored tarballs | **Done 2026-09-30**: `kaggle/vlm_eval` removed; `kaggle/w1_train` stays (trainer for the shipped reader) |
| 4 | `tv_refinement.py` — runtime-unreachable: only `vector_raster_hybrid.hybrid_vector_raster_upscale()` imports it, and no runtime code calls that function | `vector_raster_hybrid.py:584`; grep callers | **Done 2026-09-30**: out of the shipped module lists; kept for tests/evals |
| 5 | **Fidelity mode was dead code:** it imported `drunet` + `deep_unfolding` (legacy-only) against a checkpoint that never shipped | `smart_upscaler.py` (removed) | **Done 2026-09-30**: mode removed; nothing else referenced it |
| 6 | **`rrdbnet` module was not shipped** — the x4plus `.pth` path does `from rrdbnet import RRDBNet`, and the module existed only under `legacy/research/` | `sr_engine.py:74` | **Done 2026-09-30**: `rrdbnet.py` moved to the root; wheel + PyInstaller lists updated |
| 7 | `weights/onnx/x4v3_fp16.onnx` — referenced by no code | grep `x4v3_fp16` = 0 hits | **Done 2026-09-30**: local file deleted (weights are not tracked) |
| 8 | `deva_reader.py` + `deva_crnn/` were not in the wheel/PyInstaller lists, so the documented `--deva-lines` recipe could not work from a build | paths above | **Done 2026-09-30**: shipped in wheel + PyInstaller lists; the web image stays document-only by design (no CLI there) |
| 9 | `.dockerignore` excluded `calibration/` while the code shipped → `cal_conf` was silently inert in the image | `tests/test_document_flags.py:84-85` | **Done 2026-09-30**: the image ships `calibration/` |
| 10 | `cli.py --workers` was declared and never read | `cli.py` (removed) | **Done 2026-09-30**: flag and docstring claim removed |
| 11 | CLI grain is a flat per-pixel formula; `SmartUpscaler` grain is luminance-conditioned — same flag, two looks | `cli.py` vs `smart_upscaler.py` | **Held for P3**: the web photo studio will use `SmartUpscaler`; the CLI keeps its own flag |
| 12 | Runtime list carried dev-only deps: `torchvision`, `onnx`, `transformers` (legacy/eval only). `scipy` looked dev-only but **stays** (scikit-image imports it transitively); `torch`/`huggingface_hub` back the photo/verifier features | import audit + the blocked-import check | **Done 2026-09-30**: split into `requirements.txt` (full), `requirements-doc.txt` (lean web/CLI-document), dev extras |
| 13 | Local disk weight (git-ignored): `data/` 4.55 GB, `out/` 1.47 GB, `deva_crnn/` 1.16 GB, `upscayl-repo/` 231 MB, `weights/` 116 MB, `node_modules/` 109 MB, `brag-output/` 69 MB, `webapp/runs` 28 MB, `webapp/shots` 17 MB, `web_outputs/` 17 MB. The distributed repo itself is only ~11.8 MB (`.git`) / ~3.8 MiB tracked | measured 2026-09-30 | **The GitHub repo is already lean** — the weight is local data and generated output; clean the generated part, keep user data (P0) |
| 14 | `brag-output/` is untracked *and* un-ignored; `scripts/build_digit_crop_sheet.py` is untracked though functional; `webapp/shots/` (26 screenshots) is referenced by no code | `git status`; grep | gitignore `brag-output/`; track the script; delete `shots/` local (P0) |
| 15 | Docs disagree with code: `docs/ARCHITECTURE.md:68` calls the line reader "not adopted" while v1.2 shipped it opt-in; `README.md:125` says `deva_crnn/` "research, not shipped"; test counts 313/324 vs 321 actual; `docs/RELEASE.md` still says v1.1.0-era things; `docs/DEPLOY.md` has the broken deva recipe; the web landing says "No network access is used — the models ship with the app" while `ne/hi` downloads the Devanagari rec model on first use | paths above | **Truth pass** (P0) |

### 0.2 Which AI models are actually part of the product

| Model / weights | Used by | Keep? |
|---|---|---|
| RapidOCR PP-OCRv5/6 ONNX (Latin, bundled in the wheel) | document core | **Keep** — commercial-safe (Apache-2.0) |
| RapidOCR Devanagari rec model (downloaded once on first `ne/hi` use) | document `ne/hi` | **Keep** — pre-warm or document the one-time download |
| `weights/RealESRGAN_x4plus.pth` (RRDBNet) | photo x4plus | **Keep locally** (needs fix #6); weights are CC-BY-NC-SA per the repo gate → never in hosted/commercial builds |
| `weights/realesr-general-x4v3.pth` (SRVGGNetCompact, `srvggnet.py`) | photo x4v3 / auto | **Keep** — code ships |
| `weights/onnx/x4plus_fp16.onnx` | photo auto (no arch module needed) | **Keep** |
| `weights/onnx/x4v3_fp16.onnx` | nothing | **Delete** |
| ncnn UltraSharp / Remacri / High-Fidelity + `upscayl-bin.exe` (`upscayl-repo/`, AGPL + non-commercial labels) | photo, optional local | **Never in the product path** — keep out of web/hosted, no code change needed |
| `weights/deva_crnn_h48w512.pt` | `--deva-lines` only (off by default) | Keep — but fix shipping (#8) or drop the feature |
| bodhan `indic-ocr` digit verifier (~1.9 GB, gated repo, self-host license only) | `--digit-verifier bodhan` only | **Keep as opt-in CLI**; never auto-download in web/startup |
| `drunet`/`deep_unfolding` deep-SR checkpoint (never published) | "fidelity" mode | **Not a product model** — remove the mode or publish the checkpoint (#5) |
| Nepali lexicon (`scripts/fetch_nepali_lexicon.py`), calibration JSON | optional `unknown_word`, `cal_conf` | Keep; fix Docker calibration (#9) |

**Bottom line:** no model is being downloaded or bundled that shouldn't be;
the real waste was *code that can never run* (legacy, kaggle/vlm_eval,
fidelity, tv_refinement, a dead ONNX file — all removed 2026-09-30),
*deps that should be dev-only* (split 2026-09-30), and *the untracked web UI*
(committed 2026-09-30).

### 0.3 Where the web stands vs. the CLI (the parity target)

The gap is almost entirely API fields + controls — `run_document_pipeline`
already has every document knob. The photo studio is the only large lift, and
it is license-gated. Current web: language (default `ne`) + deskew, first PDF
page only, fixed artifact set (`webapp/vvweb/api.py:109-123`,
`webapp/vvweb/pipeline.py:33-69`, `webapp/frontend/src/lib/studio-state.ts:11`).
Notable details: the startup warm-up runs with `lang=None` (`api.py:65`) while
the UI defaults to `ne` — so the first visitor pays the Devanagari model
download; `meta.resized` is returned but not shown. (Until 2026-09-30 the
comparison point was `app.py`, the Gradio studio; it was removed by user
decision — the web app is now the only UI and the CLI keeps every feature.)

---

## 1. Feature-parity plan (matrix)

Legend: **A** available · **P** partial · **—** missing. "Gap fix" = what the
web needs; all pipeline kwargs already exist unless noted.

| Capability | Gradio | CLI | Web today | Gap fix | Phase |
|---|---|---|---|---|---|
| Image upload, drag/drop/paste, EXIF | A | A | A | — | — |
| PDF input, first page | A | A | A | — | — |
| Multi-page PDF + combined PDF/TXT | A | A (`--max-pages 0`) | — | `all_pages` form field + loop + `write_searchable_pdf_pages` + page cap | P2 |
| OCR engine choice (auto/rapidocr/tesseract) | A | A | — | `ocr` form field | P2 |
| Language en/ne/hi | A (default en) | A | A (default ne) | align defaults; pre-warm the configured default | P2/P4 |
| Deskew | A | A | A | — | — |
| Auto-rotate toggle | implicit on | A | implicit on | optional toggle (`auto_rotate`) | P2 |
| Output toggles (overlay/pdf/txt) | A | A | always on | form booleans → `make_*` | P2 |
| Downscale + device notes | A | P | P (`meta.resized` ignored) | render `resized`, add `device` to meta | P2 |
| Review queue UI (flags, alt readings, risk) | P | P | **A** (best of the three) | — | — |
| Markdown transcript export (transcript + review queue) | (removed) | A (`<stem>.md`) | A (`transcript.md`) | — | **done 2026-09-30** |
| Digit re-pass policy | implicit (language policy) | A toggle | implicit | optional toggle (`repass_digits`) | P2 |
| Mixed router (opt-in) | — | A | — | form bool + toggle, with the honesty note | P2 |
| Deva line reader (opt-in) | — | A | — | form enum + server ckpt path (after #8) | P2/P3 |
| Digit verifier (bodhan) | — | A | — | **not exposed** — 1.9 GB, self-host license; CLI only | — |
| Photo: scale 2/4, mode, model, TTA, grain, SVG, mask | A | P | — | new `/api/upscale` + route, gated + capped | P3 |
| Batch folders, report, skip-existing | — | A | — | **not planned** — CLI workflow | — |
| Rate limit, TTL, auth, queue shedding | — | — | **A** | keep; share the worker with photo | P3 |
| Device/model info in health | P | P | — | extend `/api/health` (`features`, `device`, model availability) | P2/P3 |

> The Gradio column is historical: `app.py` was removed on 2026-09-30 (user
> decision — one UI). The Web column is the product target; the CLI keeps
> every feature, including the photo stack and the batch workflow.

---

## 2. Phases

Each phase is independently shippable; sizes are S (hours) / M (a few days) /
L (a week or more) for one focused session.

### P0 — Repo truth & hygiene (S) — no behavior change

1. Track `webapp/` in git (it is the product); add `brag-output/` to
   `.gitignore`; track `scripts/build_digit_crop_sheet.py`.
2. Delete generated local cruft (never user data): `webapp/shots/`,
   `webapp/runs/*`, root `test_*.png`; offer a `scripts/clean_local.py`
   that reports before deleting anything.
3. CI: frontend build job (Node ≥22.12) + `vvweb` API tests — **done
   2026-09-30** (`engines` added; `tests/test_webapp_api.py`).
4. Docs truth pass: `docs/ARCHITECTURE.md` module map (+`deva_reader.py`,
   +`webapp/`, line-reader status), `README.md` (test count, line reader,
   web app section, `docs/PLAN_WEB_FULL.md` link), `docs/RELEASE.md`
   (version, counts), `docs/DEPLOY.md` (web app deploy + corrected deva
   recipe), landing copy honesty (`ne/hi` first-run download).

**Acceptance:** `git status` shows no unexpected untracked product paths;
CI green (tests + license + frontend build); docs statements match code.

### P1 — Diet & broken-path fixes (S–M)

1. Remove `legacy/`, `kaggle/` from the tree (git history keeps them;
   CHANGELOG note with the reason).
2. Resolve the unreachable/broken items: `tv_refinement` out of the shipped
   module list; fidelity mode removed (or checkpoint published); dead
   `x4v3_fp16.onnx` deleted; `rrdbnet.py` moved to the root and added to
   `pyproject`/Docker/spec; `deva_reader` shipping decided and implemented
   consistently (wheel + Docker + spec + `.dockerignore`); Docker gains
   `calibration/`; CLI `--workers` removed; grain unified.
3. Dependency split, without breaking `pip install -r requirements.txt`:
   - `requirements.txt` — full local product (unchanged UX).
   - `requirements-doc.txt` — lean document-only set (no torch, no gradio,
     no transformers/huggingface_hub; keeps numpy, opencv, scikit-image,
     onnxruntime, Pillow, rapidocr, reportlab, pypdfium2, jiwer).
   - `pyproject` extras: `doc`, `photo`, `verifier`, `web`, `dev`;
     `scipy`/`torchvision`/`onnx`/`transformers` leave the runtime list.
4. Keep the wheel honest: `py-modules` matches what actually ships;
   license gate rerun (`scripts/license_report.py --check`).

**Acceptance:** a fresh venv with `requirements-doc.txt` runs the CLI
document mode and the web app document path (no torch imported); the full
install runs the photo tab; suite green; license gate green.

### P2 — Web document parity (M)

1. API (`webapp/vvweb/api.py`, `pipeline.py`): add `ocr`, `overlay`, `pdf`,
   `txt`, `all_pages`, `auto_rotate` form fields; thread them to
   `run_document_pipeline`; multi-page loop with `write_searchable_pdf_pages`;
   new artifacts `combined.pdf`/`combined.txt` in `PUBLIC_FILES` and the
   manifest; extend `meta` with `resized`, `device`, `pages`, and echo `lang`.
2. Caps: page cap per run (suggest 10) + existing 180 s timeout kept
   per page; document the rule in `/api/health`.
3. Frontend (`StudioPanel`, `StudioResults`, `studio-state`, `api.ts`):
   advanced disclosure with the new controls; multi-page results (page-1
   preview + combined downloads); render `resized` note; surface
   `flags_summary`; keep `ne` default but make the Devanagari first-run
   behavior visible (one-time download hint).
4. Tests: first `vvweb` coverage — success path (stubbed pipeline),
   validation rejects, all-pages artifacts, rate limit, TTL prune.

**Acceptance:** every document row of the §1 matrix marked A; tests +
frontend build green in CI.

### P3 — Web photo studio, gated (L)

1. Config: `VECTOVECTO_WEB_PHOTO=1` opt-in (default off; `/api/health`
   advertises `features`). Document the licensing gate: the photo stack uses
   CC-BY-NC-SA weights and must not be enabled in commercial/hosted builds.
2. API `/api/upscale`: file, `scale` (2/4), `mode` (auto/photo/vector),
   `model` (auto, x4plus, x4v3, onnx when present — no ncnn/AGPL),
   `fast`, `grain`, `svg`, `mask`; artifacts `upscaled.png`, `vector.svg`,
   `mask.png`, `meta.json`; caps: input ≤ 4 MP, output ≤ 64 MP, SVG ≤ 20 MB,
   shared worker semaphore with the document path, 300 s timeout.
3. Efficiency: `import smart_upscaler` lazily inside the handler; engine
   cache keyed by spec (same pattern as `app.py:_ENGINE_CACHE`); warm only
   when photo is enabled; document-only servers never import torch.
4. Frontend: lazy `/upscale` route; controls mirror the Gradio photo tab;
   comparison = browser-scaled original vs server result; downloads; SVG and
   semantic map.
5. Weights: read from local `weights/` only (no downloads); `auto` fallback
   to bicubic surfaced honestly in `meta`.

**Acceptance:** with photo enabled + weights present, upload → 4× → downloads
work end to end; with the default config the endpoint is disabled and torch
is not imported at startup.

### P4 — Efficiency & deploy (M)

1. Docker: web image (Node build stage → `dist/`, then uvicorn) — **done
   2026-09-30**: the root `Dockerfile` is the web image; CI builds it and
   checks `/api/health`; calibration included.
2. Warm-up correctness: warm the configured default language (`ne` for the
   web UI) instead of `lang=None`; photo engine warmed only if enabled.
3. Offline guarantee, honestly: optional image layer that pre-downloads the
   Devanagari model (baked cache), otherwise document the one-time download.
4. Docs: rewrite `webapp/README.md` (features, knobs, deploy, license gate),
   add the web app to the root README quickstart.

**Acceptance:** `docker build` + `curl /api/health` in CI; fresh-clone
quickstart for (a) CLI, (b) Gradio, (c) web, each verifiable in minutes.

### P5 — Verification (S)

Run the acceptance checklist below end to end; refresh the affected numbers
in `docs/EVALUATION.md` pointers and this doc's status. Optional (not
committed): browser smoke via the existing tooling for the two studio pages.

---

## 3. Efficiency budget (what the web will refuse to do)

| Resource | Document | Photo (opt-in) |
|---|---|---|
| Upload | ≤ 12 MB, ≤ 30 MP, streamed cap | ≤ 12 MB, ≤ 4 MP |
| Pages per run | ≤ 10 (new), 180 s/page timeout | 1 |
| Output guard | existing export path | ≤ 64 MP, SVG ≤ 20 MB, 300 s |
| Worker | one heavy worker, shared queue, 503 + Retry-After | same worker |
| Models | Latin ships; Devanagari one-time download (or pre-baked) | local `weights/` only; never downloaded |
| Payloads | artifact files via endpoints (JSON stays text-small); no base64 image round-trips in API responses | same |
| Frontend | route-level code splitting; heavy libs (three.js) already lazy | own chunk |

No new service, no worker fleet, no per-request subprocess: the web app stays
a single uvicorn process — deployment simplicity is the product.

---

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Photo weights are non-commercial; enabling them on a hosted demo is a license violation | Default off; explicit env opt-in; warning in README + startup log; license table gains the ncnn/upscayl entries it currently misses |
| Devanagari first run needs network (contradicts the offline pitch if not handled) | Warm the UI default at startup; optional pre-baked image layer; honest copy |
| Multi-page PDF memory on the single worker | Page cap + per-page timeout + existing 2500 px fit |
| Removing `legacy/`//`kaggle/` breaks someone's workflow | History preserves everything; CHANGELOG note with recovery command (`git show <rev>:legacy/...`) |
| Shipping `rrdbnet.py` / `deva_reader` changes packaging | BSD-3 code, no license impact; license gate rerun in CI catches drift |
| Single-process session state (rate limit) resets on restart | Already documented; unchanged |

---

## 5. Explicit non-goals

- Batch/folder processing over the web (CLI keeps it).
- Hosting the bodhan digit verifier or auto-downloading any multi-GB model.
- ncnn/upscayl (AGPL + non-commercial labels) in the product path.
- Eval-only knobs in the UI (`primary_stream`, `conf_threshold`), except the
  ones users already see in the local studio.
- Desktop packaging (stays as prepared tooling only, per `docs/RELEASE.md`).

## 6. Acceptance checklist (end of P5)

- [ ] `git clone` → install → `python webapp/server.py` serves the studio;
      the CLI works from the same install.
- [ ] Fresh clone → web app quickstart (backend deps + `npm ci && npm run build`
      + server) works; document tab reaches the §1 matrix.
- [ ] Photo studio works locally via the web when enabled and weights exist;
      disabled by default; never imports torch in document-only mode.
- [ ] Multi-page PDF: combined PDF/TXT equals the CLI's `--max-pages 0` output.
- [ ] `python -m pytest tests -q` + license gate + frontend build + docker
      smoke all green in CI.
- [ ] No dead module in `py-modules`; every doc claim checked against code.
