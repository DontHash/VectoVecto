"""
license_report.py — P7 license gate: dependency + model license summary.

Reads the declared runtime/dev dependency sets, looks up the *installed*
distribution metadata for version and license, and writes `docs/LICENSES.md`
with a static section for models and datasets whose terms live outside
package metadata. Re-run before every release:

    python scripts/license_report.py            # writes docs/LICENSES.md
    python scripts/license_report.py --check    # exit 1 if a license is missing
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from importlib import metadata
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("RapidOCR / PP-OCRv5 det+rec+cls ONNX models", "Apache-2.0",
     "bundled with the rapidocr package; safe for commercial redistribution"),
    ("Tesseract 5 + tessdata (eng)", "Apache-2.0",
     "optional external install, not redistributed here"),
    ("UltraSharp / Real-ESRGAN weights", "CC-BY-NC-SA-4.0",
     "photo-upscale weights; NOT for commercial builds - excluded from "
     "commercial packaging"),
    ("Indic Open Model (bodhan digit verifier)", "Indic Open Model License 1.0",
     "opt-in second model; self-host allowed, no third-party hosting, "
     "attribution required; ~1.9 GB, one-time `hf auth login`"),
    ("heiDATA printed Devanagari (evaluation)", "CC BY 4.0",
     "Merkel-Hilf 2022, doi:10.11588/data/EGOKEI - manifests only, pages not "
     "redistributed"),
    ("Nepali government PDFs (evaluation)", "internal-only",
     "supremecourt.gov.np / lawcommission.gov.np; never redistributed"),
    ("himalaya-ai/nepali-deva-ocr-eval line crops (evaluation)",
     "unknown - unverified provenance",
     "provisional, behavior-only numbers until a human GT pass"),
]

# Packages whose metadata does not carry a license string.
KNOWN_LICENSES = {
    "rapidocr": "Apache-2.0",
    "onnxruntime": "MIT",
    "pypdfium2": "Apache-2.0 / BSD-3-Clause (bundled PDFium)",
    "torch": "BSD-3-Clause",
    "torchvision": "BSD-3-Clause",
    "opencv-python": "Apache-2.0",
    "numpy": "BSD-3-Clause",
    "scipy": "BSD-3-Clause",
    "scikit-image": "BSD-3-Clause",
    "pillow": "MIT-CMU",
    "gradio": "Apache-2.0",
    "reportlab": "BSD-3-Clause",
    "transformers": "Apache-2.0",
    "huggingface-hub": "Apache-2.0",
    "onnx": "Apache-2.0",
    "pytest": "MIT",
    "matplotlib": "PSF-based (matplotlib license)",
    "tqdm": "MPL-2.0 / MIT",
    "datasets": "Apache-2.0",
    "onnxconverter-common": "MIT",
    "pyiqa": "NTU S-Lab License 1.0 (research use)",
    "pyside6": "LGPL-3.0 (Qt for Python)",
    "jiwer": "Apache-2.0",
    "google-genai": "Apache-2.0",
}

_COPYRIGHT_RE = re.compile(r"^\s*Copyright", re.IGNORECASE)


def _declared(path: str) -> List[str]:
    names = []
    if not os.path.exists(path):
        return names
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        names.append(re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip())
    return names


def _license_of(dist_name: str) -> str:
    lic = ""
    try:
        meta = metadata.metadata(dist_name)
        lic = (meta.get("License") or "").strip()
        if not lic or len(lic) > 120 or _COPYRIGHT_RE.match(lic):
            for classifier in meta.get_all("Classifier") or []:
                if classifier.startswith("License ::"):
                    lic = classifier.split("::")[-1].strip()
                    break
    except metadata.PackageNotFoundError:
        pass  # CI gate job runs without the full environment installed
    if not lic:
        lic = KNOWN_LICENSES.get(dist_name.lower(), "")
    return lic or "UNKNOWN"


def _version_of(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _table(names: List[str]) -> str:
    rows = ["| Package | Version | License |", "|---|---|---|"]
    unknown = []
    for name in names:
        lic = _license_of(name)
        if lic == "UNKNOWN":
            unknown.append(name)
        rows.append(f"| `{name}` | {_version_of(name)} | {lic or 'UNKNOWN'} |")
    return "\n".join(rows), unknown


def build() -> str:
    runtime = _declared(os.path.join(BASE_DIR, "requirements.txt"))
    dev = _declared(os.path.join(BASE_DIR, "requirements-dev.txt"))
    runtime_rows, runtime_unknown = _table(runtime)
    dev_rows, dev_unknown = _table(dev)
    model_rows = "\n".join(f"| {name} | {lic} | {note} |"
                           for name, lic, note in MODELS)
    unknown = runtime_unknown + dev_unknown
    if unknown:
        gate = ("<!-- license-gate: unresolved -->\n"
                "> **Action required:** unresolved dependencies: "
                + ", ".join(f"`{u}`" for u in unknown)
                + " — resolve before a commercial release.\n")
    else:
        gate = ("<!-- license-gate: ok -->\n"
                "All declared dependencies resolved to a license string.\n")
    return f"""# Third-party licenses (P7 license gate)

Generated by `python scripts/license_report.py` - re-run before every release.
The product itself is MIT (see [../LICENSE](../LICENSE)); every third-party
component below keeps its own terms.

## Runtime dependencies

{runtime_rows}

## Development / evaluation dependencies

{dev_rows}

## Models and evaluation data

| Component | License | Notes |
|---|---|---|
{model_rows}

## Gate notes

{gate}* `PyMuPDF` (AGPL) is not a dependency; do not add it to a commercial build.
* `UltraSharp` photo weights are CC-BY-NC-SA and must stay out of commercial
  packaging; the document pipeline does not require them.
* The air-gap requirement (no network at runtime) is enforced by tests; model
  files are vendored by `rapidocr` at install time, not downloaded per run.
"""


def main():
    ap = argparse.ArgumentParser(description="Generate the license gate report")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "docs",
                                                  "LICENSES.md"))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when any license is unresolved")
    args = ap.parse_args()
    text = build()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {args.out}")
    if args.check and "license-gate: unresolved" in text:
        print("license gate: unresolved entries present")
        sys.exit(1)


if __name__ == "__main__":
    main()
