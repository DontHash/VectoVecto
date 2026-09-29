"""Selective gzip compression for text-like responses only.

Starlette's stock `GZipMiddleware` compresses every response above the size
threshold — including PNG/PDF artifacts, which are already compressed: that
costs CPU and (for streamed files) drops `Content-Length` for no size win.

This middleware reuses Starlette's responders but first checks the response
content type against an allowlist: JSON, HTML/other `text/*`, JavaScript,
XML and SVG are compressed; images, PDFs and everything else pass through
untouched.
"""
from __future__ import annotations

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

COMPRESSIBLE_PREFIXES = (
    "text/",
    "application/json",
    "application/javascript",
    "application/xml",
    "image/svg+xml",
)


def is_compressible(content_type: str) -> bool:
    """True when a response of this media type benefits from gzip."""
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    return any(ctype.startswith(prefix) for prefix in COMPRESSIBLE_PREFIXES)


class SelectiveGZipMiddleware:
    """gzip for text-like responses only; everything else is untouched."""

    def __init__(self, app: ASGIApp, minimum_size: int = 1024,
                 compresslevel: int = 6) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if "gzip" not in Headers(scope=scope).get("accept-encoding", ""):
            await self.app(scope, receive, send)
            return
        responder = _SelectiveGZipResponder(self.app, self.minimum_size,
                                            self.compresslevel)
        await responder(scope, receive, send)


class _SelectiveGZipResponder(GZipResponder):
    def __init__(self, app: ASGIApp, minimum_size: int,
                 compresslevel: int = 6) -> None:
        super().__init__(app, minimum_size, compresslevel=compresslevel)
        self._passthrough = False

    async def send_with_compression(self, message: Message) -> None:
        if self._passthrough:
            await self.send(message)
            return
        if message["type"] == "http.response.start":
            ctype = Headers(raw=message["headers"]).get("content-type", "")
            if not is_compressible(ctype):
                # Not a text-like response: never touch it (no buffering, no
                # chunked rewrite, no CPU spent gzipping compressed bytes).
                self._passthrough = True
                await self.send(message)
                return
        await super().send_with_compression(message)
