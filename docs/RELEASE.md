# Release process

## Versioning

`pyproject.toml` holds the single source of truth (`version = "X.Y.Z"`).
Tags are `vX.Y.Z` on `main`. The changelog is [../CHANGELOG.md](../CHANGELOG.md).

## Checklist

```bash
# 1. Green suite (231 tests) and license gate
python -m pytest tests/ -q
python scripts/license_report.py --check

# 2. Frozen sets unchanged since the last release
python evals/harness/eval_freeze.py --check evals/manifests/heidata_printed_v1.json
python evals/harness/eval_freeze.py --check evals/manifests/nepali_pdf_v2.json

# 3. Web image builds and answers
docker build -t vectovecto:$(python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") .

# 4. Changelog + version bump, commit, tag, push
git tag -a v1.1.0 -m "v1.1.0"
git push origin main --tags
```

## What ships in this repository

Code, tests, evaluation harnesses and frozen manifests. It deliberately does
**not** ship training data, checkpoints or third-party corpora — see
[LICENSES.md](LICENSES.md) and `.dockerignore`.

## Desktop packaging path (not built in v1.1.0)

The OSS release is CLI + web app. A signed portable Windows build stays the
Pro path; the infrastructure is prepared:

- `packaging/vectovecto.spec` — PyInstaller spec (one-folder, no console for
  the GUI entry point).
- `scripts/build_release.ps1` — installs PyInstaller, builds, zips.

Commercial builds must exclude the CC-BY-NC-SA photo weights (document mode is
self-contained). Before enabling the Pro build, re-run the license gate and
drop `smart_upscaler`/`sr_engine` from the bundle if the upscaler is not part
of the paid product.
