# legacy/ — archived photo-training scripts

These predate the document-first product (P1-P3). They are kept for
reproducibility of the photo/Phase-4-7 experiments and are **not** part of the
shipped pipeline; each script prepends the repo root to `sys.path` so it can
still import the root modules (`drunet.py`, `sr_engine.py`, ...).

Still-current photo entry points live at the repo root:
`cli.py --mode photo`, `app.py`, `export_onnx.py`, `sr_engine.py`.

Training/eval artifacts referenced by these scripts were gitignored
(`weights/`, `artifacts/`, `data/`, `out/`, `testImg/`), so most of them need
their own data to run.
