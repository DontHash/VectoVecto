"""Entry point.

    python webapp/server.py            # http://127.0.0.1:8000
    VECTOVECTO_WEB_PORT=9000 python webapp/server.py

Env: see webapp/app/config.py. Optional auth:
    VECTOVECTO_WEB_USER / VECTOVECTO_WEB_PASSWORD
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

app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
