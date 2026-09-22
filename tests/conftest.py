"""
conftest.py - import paths for the test suite.

The product modules live at the repo root (flat layout), the evaluation
harnesses under `evals/harness/`, and the data/training tools under
`scripts/`. Tests import from all three, so all three go on sys.path here;
individual test modules keep their own BASE_DIR inserts for standalone runs.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (BASE_DIR,
             os.path.join(BASE_DIR, "evals", "harness"),
             os.path.join(BASE_DIR, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
