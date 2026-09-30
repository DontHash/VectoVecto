"""
paths.py — repo-layout roots for source checkouts.

`ROOT` is the directory that holds `data/`, `weights/`, `fonts/`,
`calibration/` and `upscayl-repo/` (one level above this package), so moved
modules keep resolving the same files they did when they sat at the root.
"""
from __future__ import annotations

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)
