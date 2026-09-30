"""
branding.py — product name and environment-variable compatibility.

The product is **VeriScript**. Current deployments read `VERISCRIPT_*`
environment variables; the former name's `VECTOVECTO_*` variables keep
working as a fallback, so existing shells, Docker commands and hosted
deployments do not need to change.

No heavy imports; safe for the shipped path.
"""
from __future__ import annotations

import os

NAME = "VeriScript"
ENV_PREFIX = "VERISCRIPT_"
LEGACY_NAME = "VectoVecto"
LEGACY_ENV_PREFIX = "VECTOVECTO_"


def env(name: str, default=None):
    """`VERISCRIPT_<name>` when set, else `VECTOVECTO_<name>`, else default."""
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        value = os.environ.get(LEGACY_ENV_PREFIX + name)
    return default if value is None else value
