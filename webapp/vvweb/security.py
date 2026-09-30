"""Input validation, rate limiting, auth, and response hardening."""
from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import settings

# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

_PDF_MAGIC = b"%PDF-"

_ERROR_CODES = {
    "expired": "This result has expired. Run the page again.",
    "busy": "The demo worker is busy with another page. Try again in a moment.",
}


class UploadRejected(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def validate_upload(filename: Optional[str], content_type: Optional[str],
                    head: bytes, path: str) -> str:
    """Return the detected kind ("image" | "pdf") or raise UploadRejected.

    `head` holds the first bytes of the file (magic checks); `path` is the
    saved upload on disk (image decode + pixel budget).
    """
    name = (filename or "").strip().lower()
    ext = ("." + name.rsplit(".", 1)[-1]) if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected(
            "Unsupported file type. Upload a PNG, JPG, WEBP, TIFF, BMP or a PDF.")
    if not head:
        raise UploadRejected("The uploaded file is empty.")
    if len(head) > 0 and os.path.getsize(path) > settings.max_upload_bytes:
        raise UploadRejected(
            f"File too large. The limit is {settings.max_upload_mb} MB per page.")

    is_pdf = ext in PDF_EXTENSIONS
    if is_pdf:
        if not head.startswith(_PDF_MAGIC):
            raise UploadRejected("That file does not look like a PDF.")
        return "pdf"

    # Image: verify with Pillow before it reaches the pipeline.
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = settings.max_image_megapixels * 1_000_000 + 1_000_000
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w * h > settings.max_image_megapixels * 1_000_000:
                raise UploadRejected(
                    f"Image too large. The limit is {settings.max_image_megapixels} megapixels.")
            if w < 64 or h < 64:
                raise UploadRejected("Image too small to read (minimum 64x64).")
    except UploadRejected:
        raise
    except Exception:
        raise UploadRejected("That image could not be decoded.")
    return "image"


def new_run_id() -> str:
    return secrets.token_hex(8)


# --------------------------------------------------------------------------
# Rate limiting (in-memory sliding window; single process)
# --------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, window_s: int, max_in_window: int,
                 max_keys: int = 20_000) -> None:
        self.window_s = window_s
        self.max_in_window = max_in_window
        # Bounded so an IP spray cannot grow memory without limit.
        self.max_keys = max_keys
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.max_in_window:
            retry = int(self.window_s - (now - q[0])) + 1
            return False, max(retry, 1)
        q.append(now)
        if len(self._hits) > self.max_keys:
            self._evict()
        return True, 0

    def prune(self) -> None:
        self._evict()

    def tracked_keys(self) -> int:
        """Number of client keys currently held (bounded by `max_keys`)."""
        return len(self._hits)

    def _evict(self) -> None:
        """Drop stale keys; if still over the cap, drop the least recent."""
        now = time.monotonic()
        for key in [k for k, q in self._hits.items()
                    if not q or now - q[-1] > self.window_s]:
            self._hits.pop(key, None)
        if len(self._hits) > self.max_keys:
            by_last = sorted(self._hits.items(),
                             key=lambda kv: kv[1][-1] if kv[1] else 0.0)
            for key, _q in by_last[: len(self._hits) - self.max_keys]:
                self._hits.pop(key, None)


def client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Behind a reverse proxy this is the proxy IP unless the deployment enables
    proxy headers (uvicorn `proxy_headers=True` + `forwarded_allow_ips`, wired
    by `VERISCRIPT_WEB_FORWARDED_ALLOW_IPS`; legacy `VECTOVECTO_WEB_FORWARDED_ALLOW_IPS` also
    accepted); see docs/DEPLOY.md.
    """
    return request.client.host if request.client else "unknown"


# --------------------------------------------------------------------------
# Daily run budget (the billing kill-switch)
# --------------------------------------------------------------------------


class DailyQuota:
    """Hard per-day run budget, shared by all clients (single process).

    Counts every accepted run attempt and resets at UTC midnight. This is the
    cost ceiling that survives a distributed flood: even many IPs cannot make
    the demo process more than ``limit`` pages per day. ``limit == 0``
    disables the quota (self-hosted/private deployments).
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self._lock = threading.Lock()
        self._day = self._utc_day()
        self._used = 0

    @staticmethod
    def _utc_day() -> int:
        return int(time.time() // 86400)

    def check(self) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds); counts the run when allowed."""
        if self.limit == 0:
            return True, 0
        with self._lock:
            day = self._utc_day()
            if day != self._day:
                self._day, self._used = day, 0
            if self._used >= self.limit:
                reset = (self._day + 1) * 86400 - int(time.time())
                return False, max(reset, 1)
            self._used += 1
            return True, 0

    def state(self) -> Dict[str, int]:
        with self._lock:
            used = 0 if self._utc_day() != self._day else self._used
        return {"limit": self.limit, "used": used}


# --------------------------------------------------------------------------
# Optional basic auth
# --------------------------------------------------------------------------


def require_auth(request: Request) -> None:
    if not settings.auth_enabled:
        return
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authentication required",
                            headers={"WWW-Authenticate": 'Basic realm="VeriScript"'})
    import base64
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        user, _, password = decoded.partition(":")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Bad credentials",
                            headers={"WWW-Authenticate": 'Basic realm="VeriScript"'})
    ok = (secrets.compare_digest(user, settings.auth_user or "")
          and secrets.compare_digest(password, settings.auth_password or ""))
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Bad credentials",
                            headers={"WWW-Authenticate": 'Basic realm="VeriScript"'})


# --------------------------------------------------------------------------
# Security headers
# --------------------------------------------------------------------------

CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data: blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "media-src 'self'; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'self'"
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def cache_control_for_path(path: str) -> str | None:
    """Cache policy for the static site. Hashed build assets are immutable;
    fonts/examples cache for a week; everything else is left alone (HTML is
    handled by content type in the middleware, /api/ gets no-store)."""
    if path.startswith("/assets/"):
        return "public, max-age=31536000, immutable"
    if path.startswith(("/fonts/", "/examples/")):
        return "public, max-age=604800"
    return None


class SecurityHeadersMiddleware:
    """Security headers + cache policy on every response.

    Pure ASGI (not `BaseHTTPMiddleware`): the stock base middleware re-streams
    bodies in chunks, which would defeat the gzip middleware's minimum-size
    rule (small JSON would get compressed) and adds a needless buffering hop.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive,
                       send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)
                path = scope.get("path", "")
                if path.startswith("/api/"):
                    headers["Cache-Control"] = "no-store"
                elif message["status"] == 200:
                    if "text/html" in headers.get("content-type", ""):
                        headers.setdefault("Cache-Control", "no-cache")
                    else:
                        policy = cache_control_for_path(path)
                        if policy:
                            headers.setdefault("Cache-Control", policy)
            await send(message)

        await self.app(scope, receive, send_with_headers)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    headers = dict(exc.headers or {})
    return JSONResponse(status_code=exc.status_code, content={"error": detail},
                        headers=headers)
