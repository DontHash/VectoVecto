# -*- mode: python ; coding: utf-8 -*-
"""
vectovecto.spec — PyInstaller spec for the Pro desktop build (not built in
v1.1.0; the OSS release is CLI + web app).

Two executables from one tree:
  * vectovecto        console  -> cli.py      (document/photo modes)
  * vectovecto-studio windowed -> app.py      (Gradio studio)

Commercial builds must exclude the CC-BY-NC-SA photo weights: build with
`--exclude-module smart_upscaler` (and drop the photo tab via
VECTOVECTO_DOCUMENT_ONLY=1) unless the upscaler is part of the paid product.

Known caveats:
  * gradio and rapidocr pull data files dynamically — the collect_all() calls
    below are the minimum; verify the built app on a clean machine.
  * onnxruntime ships shared libraries; keep the one-folder mode (COLLECT),
    one-file is possible but slow to start.

Build:
    pyinstaller packaging/vectovecto.spec --noconfirm
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent.parent

datas, binaries, hiddenimports = [], [], []
for pkg in ("rapidocr", "onnxruntime", "gradio", "reportlab", "pypdfium2"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # package missing in the build env
        print(f"[spec] skip {pkg}: {exc}")

PRODUCT_MODULES = [
    "app", "calibration", "cli", "degradation_document", "doc_data",
    "doc_metrics", "document_export", "document_layout", "document_ocr",
    "document_orientation", "document_pipeline", "document_restore",
    "document_router", "document_verifier", "smart_upscaler", "sr_engine",
    "srvggnet", "tv_refinement", "vector_raster_hybrid",
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
    name="vectovecto",
    console=True,
    upx=False,
)

a_gui = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + PRODUCT_MODULES,
    excludes=["matplotlib", "pytest"],
    noarchive=False,
)
pyz_gui = PYZ(a_gui.pure)

exe_gui = EXE(
    pyz_gui, a_gui.scripts, [],
    exclude_binaries=True,
    name="vectovecto-studio",
    console=False,
    upx=False,
)

coll = COLLECT(
    exe_cli, a_cli.binaries, a_cli.datas,
    exe_gui, a_gui.binaries, a_gui.datas,
    strip=False,
    upx=False,
    name="vectovecto",
)
