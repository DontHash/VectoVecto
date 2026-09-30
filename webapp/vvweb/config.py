"""Web app configuration (env-driven).

Reads `VERISCRIPT_*` first; the legacy `VECTOVECTO_*` names still work as a
fallback (see `branding.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from veriscript import branding


def _env_int(name: str, default: int) -> int:
    try:
        return int(branding.env(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """All runtime knobs in one place. Env-first; safe defaults."""

    host: str = field(default_factory=lambda: branding.env("WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("WEB_PORT", 8000))

    # Upload limits
    max_upload_mb: int = field(default_factory=lambda: _env_int("WEB_MAX_UPLOAD_MB", 12))
    max_image_megapixels: int = field(default_factory=lambda: _env_int("WEB_MAX_MP", 30))

    # Processing
    process_timeout_s: int = field(default_factory=lambda: _env_int("WEB_TIMEOUT_S", 180))
    max_concurrent: int = field(default_factory=lambda: _env_int("WEB_CONCURRENCY", 1))
    queue_wait_s: int = field(default_factory=lambda: _env_int("WEB_QUEUE_WAIT_S", 25))

    # OCR language warmed at startup so the first visitor does not pay the
    # model init / one-time Devanagari download (the UI defaults to Nepali).
    warm_lang: str = field(default_factory=lambda: branding.env("WEB_WARM_LANG", "ne"))

    # Trusted reverse proxy for real client IPs (rate limiting). Uvicorn
    # `proxy_headers` + `forwarded_allow_ips`; empty = off. See docs/DEPLOY.md.
    forwarded_allow_ips: str = field(default_factory=lambda: str(
        branding.env("WEB_FORWARDED_ALLOW_IPS", "")).strip())

    # Retention
    run_ttl_minutes: int = field(default_factory=lambda: _env_int("WEB_TTL_MINUTES", 60))
    prune_interval_s: int = field(default_factory=lambda: _env_int("WEB_PRUNE_S", 600))

    # Rate limiting (per client IP, sliding window)
    rate_window_s: int = field(default_factory=lambda: _env_int("WEB_RATE_WINDOW_S", 120))
    rate_max_in_window: int = field(default_factory=lambda: _env_int("WEB_RATE_MAX", 6))

    # Optional basic auth for the API (set both to enable)
    auth_user: str | None = field(default_factory=lambda: branding.env("WEB_USER") or None)
    auth_password: str | None = field(default_factory=lambda: branding.env("WEB_PASSWORD") or None)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_user and self.auth_password)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def run_ttl_seconds(self) -> int:
        return self.run_ttl_minutes * 60


settings = Settings()
