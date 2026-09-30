"""
test_logging_setup.py — one logging configuration for CLI/web/library (W5).

Server-side failures must be logged (`vectovecto.web` — internal logger name,
kept from the former brand — in the web API is covered by
`tests/test_webapp_api.py`); `configure_logging()` is idempotent and reads
`VERISCRIPT_LOG_LEVEL` with the legacy `VECTOVECTO_LOG_LEVEL` as fallback.
"""
from __future__ import annotations

import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logging_setup import configure_logging, get_logger  # noqa: E402


def test_get_logger_prefixes_root():
    assert get_logger("web").name == "vectovecto.web"
    assert get_logger("vectovecto.web").name == "vectovecto.web"


def test_configure_logging_level_from_env(monkeypatch):
    monkeypatch.setenv("VERISCRIPT_LOG_LEVEL", "debug")
    assert configure_logging().level == logging.DEBUG
    monkeypatch.setenv("VERISCRIPT_LOG_LEVEL", "not-a-level")
    assert configure_logging().level == logging.INFO


def test_configure_logging_legacy_env_alias(monkeypatch):
    monkeypatch.delenv("VERISCRIPT_LOG_LEVEL", raising=False)
    monkeypatch.setenv("VECTOVECTO_LOG_LEVEL", "warning")
    assert configure_logging().level == logging.WARNING


def test_configure_logging_is_idempotent():
    logger = configure_logging()
    count = len(logger.handlers)
    configure_logging()
    configure_logging()
    assert len(logger.handlers) == count
