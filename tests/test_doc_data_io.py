"""
test_doc_data_io.py — Unicode-safe image IO.

Regression for the mojibake bug: cv2.imwrite on a Devanagari path created a
differently-named file (UTF-8 bytes reinterpreted as cp1252), so os.path.exists
and eval_freeze (hash/size checks) could not see the pages the eval had just
scored. imwrite_safe/imread_safe keep one true name.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_data import imread_safe, imwrite_safe  # noqa: E402


def test_unicode_filename_roundtrip(tmp_path):
    name = "आर्थिक विधेयक, २०८३_p000_clean.png"
    path = str(tmp_path / name)
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    assert imwrite_safe(path, img)
    assert os.path.exists(path), "file must exist under its real (Unicode) name"
    assert os.listdir(tmp_path) == [name], "no mojibake sibling may appear"
    back = imread_safe(path)
    assert back is not None and back.shape == img.shape
    assert int(back[:, :, 1].mean()) == 200


def test_imread_safe_missing_and_garbage(tmp_path):
    assert imread_safe(str(tmp_path / "nope_.png")) is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert imread_safe(str(bad)) is None
