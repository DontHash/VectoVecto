"""Input validation, rate limiting, auth, and response hardening."""
from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

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
    def __init__(self, window_s: int, max_in_window: int) -> None:
        self.window_s = window_s
        self.max_in_window = max_in_window
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
        return True, 0

    def prune(self) -> None:
        now = time.monotonic()
        for key in list(self._hits.keys()):
            q = self._hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if not q:
                del self._hits[key]


def client_key(request: Request) -> str:
    """Best-effort client identity. Behind a proxy this is the proxy IP unless
    the deployment sets X-Forwarded-For handling; documented in README."""
    return request.client.host if request.client else "unknown"


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
                            headers={"WWW-Authenticate": 'Basic realm="VectoVecto"'})
    import base64
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        user, _, password = decoded.partition(":")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Bad credentials",
                            headers={"WWW-Authenticate": 'Basic realm="VectoVecto"'})
    ok = (secrets.compare_digest(user, settings.auth_user or "")
          and secrets.compare_digest(password, settings.auth_password or ""))
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Bad credentials",
                            headers={"WWW-Authenticate": 'Basic realm="VectoVecto"'})


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


async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    headers = dict(exc.headers or {})
    return JSONResponse(status_code=exc.status_code, content={"error": detail},
                        headers=headers)
