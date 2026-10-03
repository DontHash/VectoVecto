# Packaging plan — Python distribution (and the Node SDK question)

Status: **P0 implemented 2026-10-03** (data shipping, extras split, wheel
smoke in CI). P1 (release plumbing) and P2 (Node client) are planned, not
started. Feasibility was tested by building and installing the wheel, not by
inspection alone.

## 1. Verdict

**Python package: feasible now, ~1.5 days to publish.** The wheel already
built and ran the document pipeline end-to-end from a clean venv. The gaps
were data shipping, dependency weight and release plumbing — not structure.

**Node package: a different artifact.** There is no Node library in this
repo (`webapp/frontend` is a private app) and the pipeline cannot be ported
to JS (RapidOCR/ONNX + reportlab). The useful npm artifact is a typed client
for the HTTP API (9 routes, OpenAPI served at `/openapi.json`), not an OCR
CLI wrapper.

## 2. What P0 changed

| Area | Change |
|---|---|
| Calibration data | `calibration/rapidocr_devanagari_v1.json` ships via wheel data-files; `resolve_calibration()` probes the repo copy then `<sys.prefix>/calibration/` |
| Lexicon | stays out of the wheel (2.9 MB, `data/` is gitignored by policy); `resolve_path()` probes `<sys.prefix>/lexicon/` if a distribution bundles it; absence is already graceful |
| Fonts | unchanged (already shipped via data-files; `_FONT_CANDIDATES` probes `sys.prefix`) |
| Dependencies | `torch` and `huggingface_hub` moved out of the core set into extras: `photo` (torch) and `verifier` (torch + transformers + huggingface_hub); `dev` keeps torch for tests |
| CI | new `wheel` job: builds `python -m build --wheel`, installs it into a clean venv **without** the photo/verifier extras, asserts calibration + font resolve and `veriscript --help` answers |

Verified locally:

- `veriscript-1.4.0-py3-none-any.whl` builds; contents include
  `veriscript-1.4.0.data/data/calibration/rapidocr_devanagari_v1.json` and
  `.../fonts/Mukta-Regular.ttf`.
- Clean-venv install (deps from the doc set only): `import veriscript` from
  `site-packages`, `veriscript --help` works, a full `--mode document` run on
  the sample invoice writes PDF/txt/md/JSON/overlay.
- `load_calibration()` and `unicode_font_path()` resolve from the installed
  data; `load_lexicon()` is absent-but-graceful.

## 3. P1 — release plumbing (planned)

1. Version/changelog discipline: cut `[Unreleased]` into a version, bump
   `pyproject.toml`, tag.
2. Build `sdist` + wheel; Publish to **TestPyPI** and run the same smoke on
   the published artifact (`pip install --index-url test.pypi.org/...`).
3. PyPI **Trusted Publishing** (no API tokens) from a `release` workflow
   triggered by tag push.
4. README/package metadata: add the PyPI install line and extras once live.
5. Optional: ship the lexicon as a separately-fetched extra or a
   `veriscript fetch-lexicon` command (licenses are clean; the policy
   decision is whether to distribute the 2.9 MB file on PyPI).

## 4. P2 — Node client (planned)

- `packages/veriscript-client`: TypeScript, ESM, `fetch`-based, zero runtime
  deps (Node >= 18). Surface: `health`, `restore` (multipart upload +
  polling), `files`, `correct`, `exportCorrections`, `memory`.
- Tests: vitest against a mocked fetch; one optional integration test
  against a locally started `webapp/server.py`.
- Publish scoped to npm (name TBD: `@veriscript/client` or similar); document
  the hosted demo's rate/quota limits (429 + `Retry-After` handling is part
  of the client).
- Explicit non-goal: no local OCR in Node; the client targets a VeriScript
  server (hosted or self-hosted).

## 5. Open decisions

- Lexicon on PyPI: ship it, or keep it a post-install fetch?
- npm scope/name and whether the client is versioned with the Python package
  or independently.
- Whether `deva_crnn` (currently in the wheel for reader inference) should
  stay in the product wheel long-term or move to its own distribution.
