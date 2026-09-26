"""
test_logging_setup.py — one logging configuration for CLI/studio/library (W5).

Server-side failures must be logged (container logs), not only rendered into
the Gradio status markdown; `configure_logging()` is idempotent and reads
`VECTOVECTO_LOG_LEVEL`.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logging_setup import configure_logging, get_logger  # noqa: E402


def test_get_logger_prefixes_root():
    assert get_logger("app").name == "vectovecto.app"
    assert get_logger("vectovecto.app").name == "vectovecto.app"


def test_configure_logging_level_from_env(monkeypatch):
    monkeypatch.setenv("VECTOVECTO_LOG_LEVEL", "debug")
    assert configure_logging().level == logging.DEBUG
    monkeypatch.setenv("VECTOVECTO_LOG_LEVEL", "not-a-level")
    assert configure_logging().level == logging.INFO


def test_configure_logging_is_idempotent():
    logger = configure_logging()
    count = len(logger.handlers)
    configure_logging()
    configure_logging()
    assert len(logger.handlers) == count


def test_app_document_failure_is_logged(caplog, monkeypatch):
    import app
    import document_pipeline

    def boom(*_args, **_kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(document_pipeline, "run_document_pipeline", boom)
    img = np.full((60, 120, 3), 255, np.uint8)
    with caplog.at_level(logging.ERROR, logger="vectovecto.app"):
        out = app.process_document(img, None)

    status = out[-1]
    assert "pipeline failed" in status and "pipeline exploded" in status
    errors = [r for r in caplog.records
              if r.name == "vectovecto.app" and r.levelno == logging.ERROR]
    assert errors, "the server side must log the failure"
    assert "pipeline failed" in errors[0].getMessage()
    assert errors[0].exc_info and "pipeline exploded" in str(errors[0].exc_info[1])


def test_app_unreadable_input_is_logged(caplog, monkeypatch):
    import app
    import document_orientation

    monkeypatch.setattr(document_orientation, "load_image_bgr",
                        lambda _path: None)
    with caplog.at_level(logging.WARNING, logger="vectovecto.app"):
        out = app.process_document("missing.png", None)
    assert out[-1] == "Could not read that image."
    assert any(r.name == "vectovecto.app" and r.levelno == logging.WARNING
               for r in caplog.records)
