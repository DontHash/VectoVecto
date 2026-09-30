# -*- mode: python ; coding: utf-8 -*-
"""
veriscript.spec — PyInstaller spec for the Pro desktop build (not built in
OSS releases; the OSS product is CLI + the web studio in `webapp/`).

One executable from one tree:
  * veriscript  console -> cli.py  (document/photo modes)

The studio is the web app (`python webapp/server.py`), not a desktop window;
package it separately if a desktop shell is ever wanted.

Commercial builds must exclude the CC-BY-NC-SA photo weights: build with
`--exclude-module smart_upscaler` unless the upscaler is part of the paid
product.

Known caveats:
  * rapidocr pulls data files dynamically — the collect_all() calls below are
    the minimum; verify the built app on a clean machine.
  * onnxruntime ships shared libraries; keep the one-folder mode (COLLECT),
    one-file is possible but slow to start.

Build:
    pyinstaller packaging/veriscript.spec --noconfirm
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent.parent

datas, binaries, hiddenimports = [], [], []
for pkg in ("rapidocr", "onnxruntime", "reportlab", "pypdfium2"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # package missing in the build env
        print(f"[spec] skip {pkg}: {exc}")

PRODUCT_MODULES = [
    "calibration", "cli", "degradation_document", "deva_reader", "deva_crnn",
    "doc_data", "doc_metrics", "document_export", "document_layout",
    "document_ocr", "document_orientation", "document_pipeline",
    "document_restore", "document_router", "document_verifier", "lexicon",
    "logging_setup", "branding", "rrdbnet", "smart_upscaler", "sr_engine",
    "srvggnet",
    "vector_raster_hybrid",
]

a_cli = Analysis(
    [str(ROOT / "cli.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + PRODUCT_MODULES,
    excludes=["matplotlib", "pytest", "tkinter"],
    noarchive=False,
)
pyz_cli = PYZ(a_cli.pure)

exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True,
    name="veriscript",
    console=True,
    upx=False,
)

coll = COLLECT(
    exe_cli, a_cli.binaries, a_cli.datas,
    strip=False,
    upx=False,
    name="veriscript",
)
