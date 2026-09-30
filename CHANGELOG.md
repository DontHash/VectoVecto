# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[semantic](https://semver.org/).

## [Unreleased]

### Added

- **Markdown output**: every document run writes `<stem>.md` (web:
  `transcript.md`) — the reading-order transcript plus the review queue as a
  table; CLI knob `--no-md`, download link in the web studio.
- README: head-to-head comparison matrix on the frozen hard-10 (CER / median /
  bagCER / speed / runs-on-CPU, with metric definitions) and explicit
  PDF → Markdown usage + real output filenames; sources in
  `evals/bakeoff_results.md`.
- CI: a frontend build job (Node 22) and the first `webapp` tests
  (`tests/test_webapp_api.py`).
- **Web document parity (P2)**: the studio exposes the OCR engine choice
  (auto/rapidocr/tesseract), output toggles (overlay / PDF / transcript /
  Markdown), an auto-rotate toggle and "all pages" for PDFs (cap 10) with
  combined `combined.pdf` / `combined.txt` / `combined.md` outputs; the
  results view shows the page count, the downscale note and the combined
  downloads.
- **Hosting hardening (cost-first)**: selective gzip for text-like responses
  only (PNG/PDF untouched); immutable cache headers for hashed assets and
  1-week for fonts/examples; real-client-IP proxy opt-in
  (`VECTOVECTO_WEB_FORWARDED_ALLOW_IPS`); the rate-limiter table is bounded
  (20k keys); run dirs/manifests get owner-only permissions on POSIX; a
  hosting hardening & cost checklist lives in `docs/DEPLOY.md`.

### Fixed

- **Web app: uploaded PDFs were routed to the image loader** because streamed
  uploads have no file extension; the validator's detected kind now decides
  the loader (`webapp/vvweb/pipeline.py::_load_page`), with a regression test.

### Changed

- **Repo restructure (cosmetic).** Root product modules moved into the
  `veriscript/` package — `core/` (dataset IO, metrics, degradation),
  `document/` (the pipeline), `deva/` (line reader), `photo/` (parked
  upscaling stack); `veriscript/paths.py` keeps `data/`, `weights/`,
  `fonts/` and `calibration/` resolution. `python cli.py` still works (root
  launcher) and `python -m veriscript` / the `veriscript` console script are
  the package entry points. Packaging (`pyproject.toml`, PyInstaller spec,
  Docker `COPY`) and the structure docs updated; four stray test-generated
  PNGs removed from the root (they were already gitignored) and the vector
  benchmark now writes them under `out/vector_raster/`. Fixed a
  pre-existing `--help` crash (`-42%` in an argparse help string).
- **Project renamed: VectoVecto → VeriScript.** Brand surfaces (README,
  docs, studio UI, package metadata, Docker tags, repo links) use the new
  name; `VERISCRIPT_*` env vars are the current prefix and `VECTOVECTO_*`
  keeps working as a fallback (`branding.py`). Internal names that would
  break compatibility are unchanged: the `vectovecto.*` logger, the `vvweb`
  package and the Kaggle dataset slugs. The legacy `vectovecto` console
  script is kept alongside `veriscript`.
- **The web app is the studio.** `app.py` (Gradio) is removed; the FastAPI +
  SolidJS app under `webapp/` is the single UI — the root `Dockerfile`
  builds and serves it (`/api/health`, port 8000), and the server-side error
  logging now lives in `webapp/vvweb/api.py` (`vectovecto.web`).
- **Scope note — document-first.** The photo upscaling stack is parked
  (CLI-only, license-gated); the planned web photo studio is deferred, see
  `docs/PLAN_WEB_FULL.md` (*Parked for later*).
- Web warm-up now warms the UI's default language
  (`VECTOVECTO_WEB_WARM_LANG`, default `ne`) so the first visitor does not
  pay the Devanagari model init; `webapp/server.py` configures the shared
  logger (`VECTOVECTO_LOG_LEVEL`), and the landing copy states the one-time
  Devanagari download instead of claiming all models ship with the app.
- `docs/DEPLOY.md`, `docs/ARCHITECTURE.md` and the READMEs describe the web
  studio; `docs/LICENSES.md` drops gradio.
- **Dependencies split**: `requirements.txt` is the full local product,
  `requirements-doc.txt` the lean document-only set (no torch) used by the
  web image; `scipy` stays (scikit-image imports it transitively —
  caught by a blocked-import check), `torchvision`/`onnx`/`transformers`
  leave the runtime list.
- `rrdbnet.py` (the x4plus RRDBNet) moved to the repo root; `deva_reader` +
  `deva_crnn` are in the wheel/PyInstaller lists, so `--deva-lines` works
  from builds (the web image stays document-only by design).

### Removed

- Gradio dependency, `vectovecto-studio` console script, `app.py` and its
  tests; the desktop spec is CLI-only.
- Fidelity (deep-SR) mode — its checkpoint was never published, so it could
  only fall back to the engine.
- `legacy/` research tree and `kaggle/vlm_eval` (git history keeps them; the
  `kaggle/w1_train` training kernel stays — see `docs/TRAINING.md`).
- `cli.py --workers` (declared, never read) and the dead local
  `x4v3_fp16.onnx` weight file.

## [1.3.0] - 2026-09-26

### Added

- **W-A: Nepali lexicon fetch** (`scripts/fetch_nepali_lexicon.py`): merges
  tesseract-ocr/langdata `nep.wordlist` (Apache-2.0) and
  `nepali-brihat-sabdakosh-json` (MIT) into a 132,997-word NFC-normalized list
  plus a sha256 manifest under `data/lexicon/` (git-ignored; not
  redistributed). Air-gapped builds pass local sources.
- **W-B: `unknown_word` review flag** (opt-in): flags Devanagari tokens whose
  words are all out-of-lexicon; digit-bearing tokens are skipped and text is
  never changed. Frozen confirmation: modern-PDF token queue R@10
  0.148 → **0.175**, P@10 +0.4pp, digit queue unchanged; letterpress neutral.
  Active only when the lexicon is present (`VECTOVECTO_LEXICON` overrides).
  Dev-tuning tool: `scripts/tune_lexicon_flag.py`; decision record:
  [docs/PLAN.md](docs/PLAN.md) Appendix T; gate row in
  [docs/EVALUATION.md](docs/EVALUATION.md).
- **W-D: real-print textbook probe** (`scripts/harvest_nepali_textbooks.py`,
  `scripts/profile_textbook_scans.py`): CDC catalogue downloads are dead
  (404 with and without session) and the MOEST eLibrary text layers are
  legacy-font mojibake (20/20 gated out), so no born-digital textbook GT set
  exists; 4 Cornell textbook scans (collection 1813/24179, no text layers)
  are profiled as an unlabeled probe — 26.3 tokens/page, 24.8% flagged, 4.0%
  `unknown_word`, 1.46 s/page median, no CER claims. Negative results and
  sources recorded in [docs/PLAN.md](docs/PLAN.md) Appendix U.
- **W-C: CC-100 corpus text experiment** (`scripts/fetch_deva_corpus.py`,
  `--corpus` in `scripts/export_training_data.py`): 100k filtered Nepali
  sentences replace the 30% word-pool branch of the synthetic mix (digit
  branches unchanged); the corpus-trained model passes the frozen gate
  (digit-exact **0.815** [0.775, 0.853] vs 0.810 [0.769, 0.847]) but every
  delta is inside the frozen-set resolution, so the shipped v8 weights stay
  ([docs/PLAN.md](docs/PLAN.md) Appendix W). Also fixes the GCS output path in
  `deva_crnn.train` (checkpoints were written to a literal `gs:/...` dir and
  never uploaded) and adds `scripts/gcp_w1_vm_startup.sh` with the DLVM
  proprietary-driver guard for GCP training.
- **W1: Devanagari searchable PDF fixed**: the invisible text layer and the
  overlay annotations now use the bundled Mukta (OFL-1.1, `fonts/`) instead of
  Helvetica/Hershey, so Nepali/Hindi pages extract as real searchable text and
  Devanagari `alt_text` renders in the review overlay. Fallback chain: bundled
  font → system fonts (Mangal/Lohit/Noto) → Helvetica with a logged warning.
  Noto Sans Devanagari was measured to lose ASCII letters in reportlab's
  subsetter, hence Mukta; extraction round-trips are pinned by tests.
- **W5: logging foundation**: `logging_setup.py` configures a `vectovecto.*`
  stderr logger (idempotent, `VECTOVECTO_LOG_LEVEL`, default INFO) from both
  entry points. Server-side pipeline failures, unreadable inputs, CLI
  per-file errors, previously swallowed audit/verifier errors and the photo
  bicubic fallback are now logged instead of only printed or hidden; the
  Gradio status output is unchanged.
- **Phase 1 VLM bake-off (Qwen3-VL)**: `qwen3vl` (fp16 8B), `qwen3vl-4b`,
  `qwen3vl-8b-4bit` and `qwen3vl-8b-4bit-rt` candidates in
  `evals/harness/bakeoff_models.py`, with a deterministic 1 Mpx pixel cap and
  a VRAM reservation for offload headroom; `--lines-source deva_real_lines`
  and median-CER + catastrophic-rate stats in the bake-off summaries. Runs:
  SPOT V100 fp16 (`scripts/gcp_vlm_8b_startup.sh`) and Kaggle T4 runtime-NF4
  (`kaggle/vlm_eval/`). Frozen result: the 8B cuts page CER 0.53 → **0.176**
  on the hardest 10 modern pages (reading order) but costs 152.7 s/page and
  invents 224 tokens; lines: median CER 0.0 yet digit-exact 0.077 (Bengali
  numerals) — **no VLM mode ships**; evidence in
  [evals/bakeoff_results.md](evals/bakeoff_results.md) and
  [docs/PLAN.md](docs/PLAN.md) Appendix X.
- **Track D: first real-scan ground truth** (`cornell_real_v1`): 16 content
  pages from 4 Cornell eCommons Nepali textbook scans (collection 1813/24179),
  rendered at 200 dpi, engine-prefilled and **human-corrected** (0.0% invalid
  Devanagari on all pages); frozen manifest with hashes. First real-scan
  numbers: pipeline page CER **0.0541** [0.036-0.076], bagCER 0.1376, digBAG
  0.2296 (prose 0.007-0.049; table-of-contents pages 0.073-0.153). The W1
  CRNN reader **hurts** real scans (0.0541 → 0.1558 with `deva_lines on`) —
  letterpress-specific, stays off. Builder:
  `scripts/build_cornell_labeling.py`; details in
  [docs/PLAN.md](docs/PLAN.md) Appendix Z.

### Fixed

- **`doc_data.load_dataset` on POSIX**: in-dir manifests written on Windows
  stored `pages\x.png`; joining on Linux produced literal-backslash paths, so
  every page silently scored zero (GCP pages bake-off). Relative paths are now
  separator-normalized and the frozen `{"path": ...}` form is accepted;
  regression test in `tests/test_doc_data_mixed.py`.

### Changed

- **Dense-table row-major order** (`document_layout.py`): textbook
  tables-of-contents (one-line rows, no big gaps) are now read row-major when
  a guarded grid is detected (≥3 columns, ≥4 y-center rows, ≥3 cells in ≥75%
  of rows). Frozen cornell_real_v1: the 4 ToC pages mean CER 0.1167 → 0.0985
  (2 were already row-major — sorter output unchanged; the 2 ordering-affected
  pages −27.3%) and the overall page CER 0.0541 → **0.0495**; v2 hard-10/41
  bit-identical; 0 dense fires on heidata/SROIE/CORD/arXiv-2col/photo-proxy/
  mixed. The pre-registered −25% clause **missed** (−15.6%) and is recorded
  with the post-hoc amendment in [docs/PLAN.md](docs/PLAN.md) Appendix AB.
- **Reader `auto` gate tightened** (`deva_reader.page_looks_letterpress`):
  `--deva-lines auto` now requires aged/letterpress paper (Otsu-paper
  saturation ≥15 and luminance ≤225) in addition to the running-text geometry
  gate, because the CRNN reader measurably hurts real modern scans
  (cornell_real_v1: 0.0541 → 0.1558 forced). Measured: `auto` engages 0/16
  cornell, 0/41 born-digital, 0/41 photo-proxy (CER exactly reader-off) and
  42/69 heidata unchanged. `on`/`off` untouched. Appendix AA.
- **Table cell-major reading order** (`document_layout.py`): a guarded grid
  path (≥4 columns, ≥4 rows, ≥4 cells/row, short cells) reads table pages
  cell-major — wrapped lines of a cell together — instead of the engine's
  line-major rows. Frozen `nepali_pdf_v2`: the 10 hardest pages drop pipeline
  page CER 0.5280 → **0.2550** (−51.7%) and the full 41 pages 0.3417 →
  0.2751, with bagCER/digBAG/invented bit-identical (order-only). The branch
  fires 0× on heidata-printed, SROIE, CORD, arXiv two-column, photo-proxy and
  mixed fixtures (byte-identical output); `reading_order_tables` telemetry.
  Evidence: [docs/PLAN.md](docs/PLAN.md) Appendix Y.

## [1.2.0] - 2026-09-22

### Added

- **Devanagari line reader (opt-in)**: `--deva-lines on` runs the trained
  CRNN+CTC recognizer on RapidOCR's line boxes. Frozen-gate digit-exact
  **0.810 [0.769, 0.847]** (bar 0.72); inside the pipeline on frozen letterpress
  pages: page CER **0.434 -> 0.253**, bagCER 0.553 -> 0.434, +0.8 s/page,
  review queue 350 flags smaller, no silent invented digits. It is off by
  default because it measurably hurts modern table pages (+18pp CER); PDF
  inputs never use it. Weights stay server-side (`VECTOVECTO_DEVA_CKPT` or
  `weights/deva_crnn_h48w512.pt`); the recipe is in `docs/TRAINING.md`.
- `digit_added` flag: any number the reader saw that the engine did not is
  flagged for review ? never silent.
- Kaggle GPU training toolchain (`kaggle/w1_train/`, config-driven private
  kernel) and GCS-aware trainer options (`--in-h/--in-w`, cosine schedule,
  `--save-best`).

### Changed

- Gate reporting now includes bootstrap 95% CIs.

### Fixed

- Line-box merging for the reader: adjacent detector fragments merge into one
  line (the recognizer is line-level) while a drawn table rule blocks the
  merge; born-digital PDFs keep the engine reading.

## [1.1.0] - 2026-09-22

The "ship it honestly" release: measured queue wins, multi-page PDF, and a
product-first repository.

### Added

- **Multi-page PDF**: `--max-pages 0` processes every page and writes
  `<stem>_combined.pdf` / `.txt`; the studio gained an "All pages (PDF)"
  checkbox (default remains the first page).
- **2× digit re-pass, default ON for Devanagari**: frozen review-queue digit
  recall@10 **0.73 → 0.88** (letterpress) and **0.15 → 0.38** (government
  PDFs) at ≈ +0.4 s/page. Latin pages keep it off.
- **`invalid_sequence` flag** (impossible combining sequence): token R@10
  +3.0pp letterpress / +8.0pp PDFs on the frozen sets.
- **Optional digit verifier** (`--digit-verifier bodhan`) with a junk-crop
  filter and 64-token cap (32 crops: 192 s → 26 s); stays opt-in because the
  cost gate failed.
- **Hosted web app path**: `Dockerfile`, env-configured host/port, optional
  basic auth, `VECTOVECTO_DOCUMENT_ONLY` to hide the non-commercial photo tab.
- **Repository/product structure**: `docs/` (architecture, evaluation,
  deployment, licensing, release, plan), `evals/harness/` for evaluation
  tooling, `scripts/` for data/training/release tools, `legacy/` for archived
  research, `pyproject.toml` with entry points, `CHANGELOG.md`, license gate.
- **Evaluation evidence**: frozen manifests + harnesses for letterpress,
  government PDFs, real line crops and a photo proxy; Gemini-2.5-Pro anchor
  workflow with a human review worksheet.
- **Devanagari line recognizer research track** (`deva_crnn/`, not adopted):
  multi-font synthetic data, scan-realistic augmentation, montage labeling of
  real lines, warm-start training, and a pre-registered adopt-if gate.

### Changed

- Brand unified to **VectoVecto** (README, studio title, package metadata).
- `--repass-digits/--no-repass-digits` documented as the Devanagari default.
- README rewritten product-first; every claim links to a frozen-set number.

### Fixed

- Pipeline coordinate space is fixed at entry (`fit_max_side`) and recorded in
  metadata, so restore/resize never shifts token boxes.
- Verifier junk-crop filter prevents watermark/URL strips from burning seconds
  per page.

### Removed

- Pre-pivot research assets from the repository tree (photo-training notebook,
  cloud-VM training scripts moved to `legacy/`, demo media, superseded roadmap).

### Known limitations

- Devanagari line recognizer attempts failed the gate (best digit-exact 0.424
  vs bar 0.75) — RapidOCR remains the engine.
- Real phone-photo field set still pending; numbers come from a labeled proxy.
- Letterpress preprocessing (sauvola) rejected: invents tokens.

## [1.0.0] - 2026-08

Initial product: local document restore with searchable PDF, overlay,
transcript and OCR JSON; Devanagari support; orientation/layout/reading-order
handling; risk-flag queue; Gradio studio + CLI; photo upscaler sharing the
same entry points.
