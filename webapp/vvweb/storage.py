"""Per-run artifact storage with TTL pruning.

Layout:
    webapp/runs/<run_id>/restored.png, overlay.png, searchable.pdf,
                         transcript.txt, ocr.json, manifest.json

Run ids are random 16-hex tokens; the file name inside a run is a fixed
whitelist, so a client can never walk the filesystem.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Dict, Optional

RUN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
PUBLIC_FILES = {
    "restored.png": {"media": "image/png", "download": "restored.png"},
    "overlay.png": {"media": "image/png", "download": "overlay.png"},
    "searchable.pdf": {"media": "application/pdf", "download": "searchable.pdf"},
    "transcript.txt": {"media": "text/plain; charset=utf-8", "download": "transcript.txt"},
    "transcript.md": {"media": "text/markdown; charset=utf-8", "download": "transcript.md"},
    "combined.pdf": {"media": "application/pdf", "download": "combined.pdf"},
    "combined.txt": {"media": "text/plain; charset=utf-8", "download": "combined.txt"},
    "combined.md": {"media": "text/markdown; charset=utf-8", "download": "combined.md"},
    "ocr.json": {"media": "application/json", "download": "ocr.json"},
}


class RunStore:
    def __init__(self, root: str, ttl_seconds: int) -> None:
        self.root = os.path.abspath(root)
        self.ttl_seconds = ttl_seconds
        os.makedirs(self.root, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------
    def run_dir(self, run_id: str) -> str:
        if not RUN_ID_RE.match(run_id):
            raise KeyError(run_id)
        return os.path.join(self.root, run_id)

    def create(self, run_id: str) -> str:
        path = self.run_dir(run_id)
        os.makedirs(path, exist_ok=False)
        return path

    def write_manifest(self, run_id: str, manifest: Dict) -> None:
        manifest = dict(manifest)
        manifest["expires_at"] = int(time.time()) + self.ttl_seconds
        with open(os.path.join(self.run_dir(run_id), "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)

    def read_manifest(self, run_id: str) -> Optional[Dict]:
        try:
            with open(os.path.join(self.run_dir(run_id), "manifest.json"),
                      encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def resolve_file(self, run_id: str, name: str) -> Optional[str]:
        info = PUBLIC_FILES.get(name)
        if not info:
            return None
        path = os.path.join(self.run_dir(run_id), name)
        if not os.path.isfile(path):
            return None
        return path

    def prune(self) -> int:
        """Delete runs past their TTL (or without a manifest). Returns count."""
        now = time.time()
        removed = 0
        try:
            entries = os.listdir(self.root)
        except OSError:
            return 0
        for entry in entries:
            path = os.path.join(self.root, entry)
            if not os.path.isdir(path) or not RUN_ID_RE.match(entry):
                continue
            manifest = self.read_manifest(entry)
            expires = (manifest or {}).get("expires_at", 0)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if now > max(expires, mtime + self.ttl_seconds):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed
