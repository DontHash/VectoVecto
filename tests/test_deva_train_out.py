"""
test_deva_train_out.py — trainer output-path resolution (Vertex/GCE pattern).
"""
from __future__ import annotations

import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from deva_crnn.train import _upload_if_gcs, resolve_out_dirs  # noqa: E402


def test_resolve_out_dirs_local():
    local, remote = resolve_out_dirs(os.path.join("out", "deva_crnn"))
    assert local == os.path.join("out", "deva_crnn")
    assert remote is None


def test_resolve_out_dirs_gcs():
    local, remote = resolve_out_dirs("gs://bucket/w1_v9/out")
    assert local == os.path.join(tempfile.gettempdir(), "deva_crnn_out")
    assert remote == "gs://bucket/w1_v9/out"
    assert not local.startswith("gs://"), "local writes must not use gs:// paths"


def test_upload_if_gcs_noop_for_local_dir(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")
    _upload_if_gcs(str(ckpt), str(tmp_path))  # must return before any upload
