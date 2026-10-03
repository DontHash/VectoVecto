# Packaging plan — Python distribution (and the Node SDK question)

Status: **P0–P2 shipped 2026-10-03.** `veriscript 1.5.0` is live on PyPI
(TestPyPI dry run verified first) with a GitHub Release; the Node client is
implemented and CI-tested, its npm publish pending. Feasibility was tested
by building and installing the wheel, not by inspection alone.

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

## 3. P1 — release plumbing (implemented 2026-10-03)

Done:

1. Version cut: `1.4.0` → `1.5.0` in `pyproject.toml`; CHANGELOG content moved
   from `[Unreleased]` to `## [1.5.0] - 2026-10-03`.
2. `sdist + wheel` build verified locally (`python -m build`), `twine check`
   passes on both, the sdist carries `calibration/` + `fonts/`, and the wheel
   built **from the sdist** installs and resolves its data from a clean venv.
3. `.github/workflows/release.yml`: tag pushes (`v*`) build → smoke → publish
   to PyPI (trusted publishing, PEP 740 attestations) → attach artifacts to a
   GitHub Release. Manual runs can dry-run against TestPyPI (`skip-existing`)
   or build only.
4. `veriscript.__version__` now comes from installed package metadata.
5. README documents package installs + extras.

Left (account-side, once, then releases are one tag away):

- Create the pending publishers — PyPI project `veriscript`, owner
  `DontHash`, repo `VeriScript`, workflow `release.yml`, environments
  `pypi` / `testpypi`.
- Dry run: Actions → Release → Run workflow → `testpypi`, then
  `pip install --index-url https://test.pypi.org/simple/ veriscript`.
- Tag a release: bump `pyproject.toml` + CHANGELOG on `main`, then
  `git tag v1.5.0 && git push origin v1.5.0`.

**Released 2026-10-03:** TestPyPI dry run (run 37111900214) passed, then
the `v1.5.0` tag published to PyPI + created the GitHub Release (run
37112093288). A clean venv installs `veriscript==1.5.0` from PyPI and
resolves the shipped calibration file.

Open item: the lexicon stays out of the wheel (2.9 MB, `data/` is gitignored
by policy). If distribution is wanted, the licenses are clean — ship it, or
add a `veriscript fetch-lexicon` command that downloads and verifies it.

## 4. P2 — Node client (implemented 2026-10-03)

`packages/veriscript-client` — TypeScript ESM, zero runtime dependencies,
Node >= 18 (or any modern browser) over the HTTP surface: `health`,
`restore` (multipart upload), `correct`, `exportCorrections`, `memory`,
`clearMemory`, `fileUrl`, `downloadFile`. Errors surface as
`VeriScriptError` with `status` and (429) `retryAfter`.

- Tests: vitest offline suite (5 tests) plus a live smoke gated on
  `VERISCRIPT_BASE_URL`; `npm run typecheck`/`build` clean; `npm pack`
  produces an 8 kB tarball. CI runs typecheck + tests + build.
- Hosted demo limits (429 + `Retry-After`, run TTL) are documented as part
  of the client contract.
- Left: the npm name/account decision (currently `veriscript-client`) and
  the first `npm publish`.
- Explicit non-goal: local OCR in Node — the client targets a VeriScript
  server (hosted or self-hosted).

## 5. Open decisions

- Lexicon on PyPI: ship it, or keep it a post-install fetch?
- npm scope/name and whether the client is versioned with the Python package
  or independently.
- Whether `deva_crnn` (currently in the wheel for reader inference) should
  stay in the product wheel long-term or move to its own distribution.
