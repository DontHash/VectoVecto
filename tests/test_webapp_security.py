"""
test_webapp_security.py — hosting hardening: selective gzip, cache policy,
rate-limiter bounds, and the proxy-header opt-in.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "webapp"))

pytest = __import__("pytest")
pytest.importorskip("fastapi", reason="webapp dependencies not installed")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from vvweb.compression import SelectiveGZipMiddleware, is_compressible  # noqa: E402
from vvweb.security import RateLimiter, cache_control_for_path  # noqa: E402


# -- gzip policy -----------------------------------------------------------

def test_is_compressible_allowlist():
    assert is_compressible("application/json")
    assert is_compressible("text/html; charset=utf-8")
    assert is_compressible("application/javascript")
    assert is_compressible("image/svg+xml")
    assert not is_compressible("image/png")
    assert not is_compressible("application/pdf")
    assert not is_compressible("")


def _mini_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=64,
                       compresslevel=6)

    @app.get("/json")
    def json_route():
        return {"text": "x" * 2000}

    @app.get("/png")
    def png_route():
        return Response(content=b"\x89PNG" + b"0" * 2000, media_type="image/png")

    @app.get("/small")
    def small_route():
        return {"ok": True}

    return app


def test_gzip_applies_to_large_text():
    client = TestClient(_mini_app())
    r = client.get("/json", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    assert r.json()["text"] == "x" * 2000


def test_gzip_skips_binary_and_small():
    client = TestClient(_mini_app())
    png = client.get("/png", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in png.headers
    assert len(png.content) == 2004

    small = client.get("/small", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in small.headers


def test_gzip_requires_the_client_to_ask():
    client = TestClient(_mini_app())
    r = client.get("/json", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in r.headers


# -- cache policy ----------------------------------------------------------

def test_cache_control_for_path():
    assert cache_control_for_path("/assets/index-abc.js") == \
        "public, max-age=31536000, immutable"
    assert cache_control_for_path("/fonts/mukta-400.ttf") == \
        "public, max-age=604800"
    assert cache_control_for_path("/examples/invoice-hero.jpg") == \
        "public, max-age=604800"
    assert cache_control_for_path("/") is None
    assert cache_control_for_path("/api/health") is None


# -- rate limiter ----------------------------------------------------------

def test_rate_limiter_window_enforced():
    rl = RateLimiter(window_s=60, max_in_window=2)
    assert rl.check("a")[0] is True
    assert rl.check("a")[0] is True
    allowed, retry = rl.check("a")
    assert allowed is False and retry >= 1


def test_rate_limiter_bounds_its_table():
    rl = RateLimiter(window_s=60, max_in_window=100, max_keys=3)
    for i in range(10):
        assert rl.check(f"ip-{i}")[0] is True
    assert rl.tracked_keys() <= 3


# -- proxy opt-in ----------------------------------------------------------

def test_uvicorn_proxy_opt_in():
    import server

    assert server._uvicorn_extra("") == {}
    extra = server._uvicorn_extra("127.0.0.1")
    assert extra == {"proxy_headers": True, "forwarded_allow_ips": "127.0.0.1"}
