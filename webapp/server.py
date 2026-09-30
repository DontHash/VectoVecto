"""Entry point.

    python webapp/server.py            # http://127.0.0.1:8000
    VERISCRIPT_WEB_PORT=9000 python webapp/server.py

Env: see webapp/vvweb/config.py (`VERISCRIPT_*`; legacy `VECTOVECTO_*` still
accepted). Optional auth:
    VERISCRIPT_WEB_USER / VERISCRIPT_WEB_PASSWORD
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for path in (REPO, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from vvweb.config import settings  # noqa: E402
from vvweb.api import create_app  # noqa: E402
from veriscript.logging_setup import configure_logging  # noqa: E402

configure_logging()
app = create_app()


def _uvicorn_extra(allow: str) -> dict:
    """Proxy-header kwargs for uvicorn, or {} when disabled.

    Opt-in via `VERISCRIPT_WEB_FORWARDED_ALLOW_IPS` = the proxy address(es)
    (`127.0.0.1`, a subnet, or `*`). Left unset, forwarded headers are
    ignored and the proxy IP is rate-limited (safe but coarse).
    """
    allow = (allow or "").strip()
    if not allow:
        return {}
    return {"proxy_headers": True, "forwarded_allow_ips": allow}


def main() -> None:
    import uvicorn
    extra = _uvicorn_extra(settings.forwarded_allow_ips)
    if extra:
        print(f"[server] proxy headers enabled (allow: "
              f"{extra['forwarded_allow_ips']})")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info",
                limit_concurrency=settings.max_connections,
                timeout_keep_alive=10, **extra)


if __name__ == "__main__":
    main()
