"""VeriScript — offline document restoration (Devanagari first)."""
from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("veriscript")
except PackageNotFoundError:  # source checkout without an installed distribution
    __version__ = "0+unknown"

__all__ = ["__version__"]
