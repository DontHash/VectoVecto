"""
logging_setup.py — one logging configuration for CLI, studio and library code.

Product modules log under the `vectovecto.*` logger (internal name, kept from
the former brand; `vectovecto.cli`, `vectovecto.web`, ...).
`configure_logging()` is idempotent, is called by the entry points, and reads
`VERISCRIPT_LOG_LEVEL` (legacy `VECTOVECTO_LOG_LEVEL` still accepted; default
INFO). Server-side errors then reach container logs instead of only the HTTP
response; the CLI keeps its user-facing prints.

No heavy imports; safe for the shipped path.
"""
from __future__ import annotations

import logging
import sys

from veriscript import branding

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
    name = (level or branding.env("LOG_LEVEL", DEFAULT_LEVEL)).upper()
    logger.setLevel(getattr(logging, name, logging.INFO))
    return logger


def get_logger(name: str) -> logging.Logger:
    """`get_logger("app")` -> the `vectovecto.app` logger."""
    if name == ROOT_NAME or name.startswith(ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
