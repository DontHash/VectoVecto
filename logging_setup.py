"""
logging_setup.py — one logging configuration for CLI, studio and library code.

Product modules log under the `vectovecto` logger (`vectovecto.app`,
`vectovecto.cli`, `vectovecto.document_export`, ...). `configure_logging()` is
idempotent, is called by the entry points, and reads `VECTOVECTO_LOG_LEVEL`
(default INFO). Server-side errors then reach container logs instead of only
the Gradio status markdown; the CLI keeps its user-facing prints.

No heavy imports; safe for the shipped path.
"""
from __future__ import annotations

import logging
import os
import sys

ROOT_NAME = "vectovecto"
DEFAULT_LEVEL = "INFO"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_CONFIGURED = False


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure the `vectovecto` logger once; safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger(ROOT_NAME)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logger.addHandler(handler)
        _CONFIGURED = True
    name = (level or os.environ.get("VECTOVECTO_LOG_LEVEL", DEFAULT_LEVEL)).upper()
    logger.setLevel(getattr(logging, name, logging.INFO))
    return logger


def get_logger(name: str) -> logging.Logger:
    """`get_logger("app")` -> the `vectovecto.app` logger."""
    if name == ROOT_NAME or name.startswith(ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
