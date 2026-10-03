"""FastAPI application: the VeriScript web studio.

- Serves the built frontend (webapp/frontend/dist) when present.
- POST /api/restore  → run the document pipeline on one uploaded page.
- GET  /api/runs/{id}/files/{name} → artifacts of a finished run.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import (Body, Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .compression import SelectiveGZipMiddleware
from .pipeline import (PAGE_CAP, PipelineError, collect_run_summary,
                       process_page)
from .security import (DailyKeyedQuota, DailyQuota, RateLimiter,
                       SecurityHeadersMiddleware, client_key,
                       http_exception_handler, new_run_id, require_auth,
                       validate_upload)
from .storage import PUBLIC_FILES, RunStore

from veriscript.logging_setup import get_logger

_LOG = get_logger("web")

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(HERE)
FRONTEND_DIST = os.path.join(WEBAPP_DIR, "frontend", "dist")
RUNS_DIR = os.path.join(WEBAPP_DIR, "runs")

MAX_LANG = {"en", "ne", "hi"}
OCR_BACKENDS = {"auto", "rapidocr", "tesseract"}


def _flag(value: str, default: bool = True) -> bool:
    text = (value or "").strip().lower()
    if text in ("1", "true", "on", "yes"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    return default


def _require_run_dir(run_id: str) -> str:
    """The run's directory, or a 404 when it expired / never existed."""
    manifest = store.read_manifest(run_id)
    try:
        run_dir = store.run_dir(run_id)
    except KeyError:
        run_dir = ""
    if manifest is None or not run_dir or not os.path.isdir(run_dir):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="This result has expired. Run the page again.")
    return run_dir


def _result_from_run(run_dir: str):
    """Rebuild the token objects from the run's canonical `ocr.json`."""
    from veriscript.document.corrections import result_from_ocr_json

    return result_from_ocr_json(os.path.join(run_dir, "ocr.json"))


def _memory_enabled() -> bool:
    from veriscript.document import memory as correction_memory

    return correction_memory.memory_enabled()


def _parse_corrections_body(body: Optional[Dict[str, Any]]):
    from veriscript.document.corrections import MAX_CORRECTIONS, load_corrections

    raw = (body or {}).get("corrections") or []
    if not isinstance(raw, list) or len(raw) > MAX_CORRECTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"corrections must be a list of at most {MAX_CORRECTIONS} entries")
    try:
        doc = load_corrections({"pages": {"*": raw}})
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return doc.pages["*"]


def _count_pages(kind: str, path: str, all_pages: bool) -> int:
    """Quota units for one request: a 10-page PDF costs 10 pages, not 1 run."""
    if kind != "pdf" or not all_pages:
        return 1
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(path)
        try:
            return max(1, min(len(doc), PAGE_CAP))
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - best-effort count; validation handles bad files
        return PAGE_CAP


store = RunStore(RUNS_DIR, settings.run_ttl_seconds)
limiter = RateLimiter(settings.rate_window_s, settings.rate_max_in_window)
quota = DailyQuota(settings.daily_runs)
page_quota = DailyQuota(settings.daily_pages)
client_pages = DailyKeyedQuota(settings.pages_per_client)
_worker_sem = threading.Semaphore(settings.max_concurrent)
_worker_busy = threading.Event()

VERSION = "0.1.0"


def _busy() -> bool:
    return _worker_busy.is_set()


async def _periodic_prune() -> None:
    while True:
        await asyncio.sleep(settings.prune_interval_s)
        store.prune()
        limiter.prune()


def _warm_lang() -> Optional[str]:
    """Language to warm at startup, mapped like a request (`en` → default)."""
    lang = (settings.warm_lang or "ne").strip().lower()
    if lang not in MAX_LANG:
        lang = "ne"
    return None if lang == "en" else lang


def _warm_pipeline() -> None:
    """Load the OCR engine in the background so the first visitor does not pay
    the model warm-up (including the one-time Devanagari model download).
    Best-effort; failures are non-fatal."""
    try:
        import tempfile

        from veriscript.core import data as doc_data
        from veriscript.document.pipeline import run_document_pipeline

        page, _gt = doc_data.render_synthetic_invoice(seed=1, dpi=110)
        with tempfile.TemporaryDirectory(prefix="vv_warm_") as td:
            run_document_pipeline(page, backend=None, lang=_warm_lang(),
                                  out_dir=td,
                                  stem="warm", make_pdf=False, make_overlay=False,
                                  make_txt=False, make_json=False)
    except Exception:  # noqa: BLE001 - warm-up only
        pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    store.prune()
    _LOG.info("warming OCR in the background (lang=%s)", _warm_lang() or "default")
    threading.Thread(target=_warm_pipeline, daemon=True, name="vv-warmup").start()
    task = asyncio.create_task(_periodic_prune())
    try:
        yield
    finally:
        task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="VeriScript Studio", version=VERSION,
                  docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=_lifespan)

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024,
                       compresslevel=6)
    app.add_exception_handler(HTTPException, http_exception_handler)

    # -- meta --------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "version": VERSION,
            "busy": _busy(),
            "limits": {
                "max_upload_mb": settings.max_upload_mb,
                "max_image_megapixels": settings.max_image_megapixels,
                "max_pages": PAGE_CAP,
                "ttl_minutes": settings.run_ttl_minutes,
                "rate_max": settings.rate_max_in_window,
                "rate_window_s": settings.rate_window_s,
                "auth": settings.auth_enabled,
                "max_connections": settings.max_connections,
                "daily_runs": quota.state()["limit"],
                "daily_used": quota.state()["used"],
                "daily_pages": page_quota.state()["limit"],
                "daily_pages_used": page_quota.state()["used"],
                "pages_per_client": settings.pages_per_client,
                "memory": _memory_enabled(),
            },
        }

    # -- restore -----------------------------------------------------------
    @app.post("/api/restore", dependencies=[Depends(require_auth)])
    async def restore(request: Request,
                      file: UploadFile = File(...),
                      lang: str = Form("ne"),
                      deskew: str = Form("0"),
                      ocr: str = Form("auto"),
                      overlay: str = Form("1"),
                      pdf: str = Form("1"),
                      txt: str = Form("1"),
                      md: str = Form("1"),
                      all_pages: str = Form("0"),
                      auto_rotate: str = Form("1")) -> JSONResponse:
        allowed, retry_after = limiter.check(client_key(request))
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit reached. Try again shortly.",
                                headers={"Retry-After": str(retry_after)})

        lang = (lang or "ne").strip().lower()
        if lang not in MAX_LANG:
            lang = "ne"
        ocr = (ocr or "auto").strip().lower()
        if ocr not in OCR_BACKENDS:
            _LOG.warning("unknown OCR engine %r; using auto", ocr)
            ocr = "auto"
        deskew_flag = _flag(deskew, default=False)
        all_pages_flag = _flag(all_pages, default=False)

        # Stream the upload to disk with a hard cap (never trust the header).
        run_id = new_run_id()
        run_dir = store.create(run_id)
        upload_path = os.path.join(run_dir, ".upload")
        size = 0
        try:
            with open(upload_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File too large. The limit is "
                                   f"{settings.max_upload_mb} MB per page.")
                    f.write(chunk)

            with open(upload_path, "rb") as f:
                head = f.read(64)
            try:
                kind = validate_upload(file.filename, file.content_type, head,
                                       upload_path)
            except HTTPException:
                raise
        except HTTPException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Upload failed.")

        # Quotas are charged in pages (a 10-page PDF costs 10), now that the
        # upload is validated. Per-visitor first so a blocked visitor does not
        # consume the global budget.
        pages = _count_pages(kind, upload_path, all_pages_flag)
        allowed, retry_after = client_pages.check(client_key(request), pages)
        if not allowed:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"You've reached the free demo limit "
                       f"({settings.pages_per_client} pages per visitor per "
                       f"day). Try again after midnight UTC.",
                headers={"Retry-After": str(retry_after)})
        allowed, retry_after = page_quota.check(pages)
        if not allowed:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The demo's daily page budget is used up. Try again "
                       "after midnight UTC.",
                headers={"Retry-After": str(retry_after)})
        allowed, retry_after = quota.check()
        if not allowed:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The demo's daily budget is used up. Try again after midnight UTC.",
                headers={"Retry-After": str(retry_after)})

        # Single heavy worker: wait briefly, then shed load.
        acquired = _worker_sem.acquire(timeout=settings.queue_wait_s)
        if not acquired:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="The demo worker is busy with another "
                                       "page. Try again in a moment.",
                                headers={"Retry-After": "10"})
        _worker_busy.set()
        try:
            loop = asyncio.get_running_loop()
            payload = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: process_page(
                    upload_path, run_dir, run_id,
                    lang=None if lang == "en" else lang, deskew=deskew_flag,
                    kind=kind, ocr=ocr,
                    make_overlay=_flag(overlay), make_pdf=_flag(pdf),
                    make_txt=_flag(txt), make_md=_flag(md),
                    all_pages=all_pages_flag,
                    auto_rotate=_flag(auto_rotate))),
                timeout=settings.process_timeout_s)
        except asyncio.TimeoutError:
            _LOG.warning("restore timed out after %.0fs (run=%s)",
                         settings.process_timeout_s, run_id)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail="Processing timed out for this page.")
        except PipelineError as exc:
            _LOG.warning("pipeline rejected the page (run=%s): %s", run_id, exc)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except Exception:
            _LOG.exception("restore failed (run=%s)", run_id)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="The pipeline failed on this page. "
                                       "Try a cleaner scan or a different file.")
        finally:
            _worker_busy.clear()
            _worker_sem.release()
            try:
                os.remove(upload_path)
            except OSError:
                pass

        store.write_manifest(run_id, {"meta": payload.get("meta", {}),
                                      "files": list(PUBLIC_FILES.keys())})
        payload["expires_in"] = settings.run_ttl_seconds
        return JSONResponse(payload)

    # -- artifacts ---------------------------------------------------------
    @app.get("/api/runs/{run_id}/files/{name}", dependencies=[Depends(require_auth)])
    async def run_file(run_id: str, name: str, download: int = 0) -> FileResponse:
        if name not in PUBLIC_FILES:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such file.")
        manifest = store.read_manifest(run_id)
        path = store.resolve_file(run_id, name)
        if manifest is None or path is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="This result has expired. Run the page again.")
        return FileResponse(
            path,
            media_type=PUBLIC_FILES[name]["media"],
            filename=PUBLIC_FILES[name]["download"] if download else None,
            headers={"Cache-Control": "private, max-age=300"},
        )

    # -- correction loop (docs/CORRECTIONS.md) ------------------------------
    @app.post("/api/runs/{run_id}/correct", dependencies=[Depends(require_auth)])
    async def correct(run_id: str, body: Dict[str, Any] = Body(...)) -> JSONResponse:
        import cv2

        from veriscript.document.corrections import (apply_and_export,
                                                     corrections_record,
                                                     write_corrections_record)

        run_dir = _require_run_dir(run_id)
        corrections = _parse_corrections_body(body)
        if not corrections:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="no corrections supplied")
        image = cv2.imread(os.path.join(run_dir, "restored.png"))
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The run image is no longer available. Run the page again.")
        result = _result_from_run(run_dir)
        snapshots = {i: (t.text, list(t.flags))
                     for i, t in enumerate(result.tokens)}
        applied = apply_and_export(
            run_dir, "corrected", image, result, corrections,
            make_pdf=True, make_overlay=False, make_txt=True,
            make_json=True, make_md=True, overlay_source=image)
        stats = applied["stats"]
        page_stem = f"page_{run_id[:8]}"
        if stats["reviewed"]:
            write_corrections_record(
                os.path.join(run_dir, "corrections.json"),
                corrections_record(page_stem, corrections, stats, run_id))
            try:  # track D2: text-only local memory (best-effort)
                from veriscript.document import memory as correction_memory
                correction_memory.record_corrections(
                    (body or {}).get("corrections") or [], snapshots,
                    engine=result.backend)
            except Exception:  # noqa: BLE001
                pass
        summary = collect_run_summary(run_dir, run_id)
        return JSONResponse({"run_id": run_id, "stats": stats, **summary})

    @app.post("/api/runs/{run_id}/corrections/export",
              dependencies=[Depends(require_auth)])
    async def corrections_export(run_id: str,
                                 body: Dict[str, Any] = Body(...)) -> JSONResponse:
        import cv2

        from veriscript.document.corrections import build_corrections_zip

        run_dir = _require_run_dir(run_id)
        corrections = _parse_corrections_body(body)
        share_raw = (body or {}).get("share") or []
        if not isinstance(share_raw, list):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="share must be a list of indices")
        try:
            share = [int(i) for i in share_raw]
        except (TypeError, ValueError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="share must be a list of indices")
        known = {c.index for c in corrections}
        if not set(share) <= known:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="share contains indices that are not in corrections")
        image = cv2.imread(os.path.join(run_dir, "restored.png"))
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The run image is no longer available. Run the page again.")
        page_stem = f"page_{run_id[:8]}"
        build_corrections_zip(os.path.join(run_dir, "corrections.zip"),
                              image, page_stem, corrections, share)
        n_shared = sum(1 for c in corrections
                       if c.index in set(share) and c.bbox is not None)
        return JSONResponse({
            "run_id": run_id,
            "file": f"/api/runs/{run_id}/files/corrections.zip",
            "shared": n_shared,
        })

    # -- local correction memory (track D2, docs/HARNESS_PLAN.md) -----------
    @app.get("/api/memory", dependencies=[Depends(require_auth)])
    async def memory_status() -> Dict[str, Any]:
        from veriscript.document import memory as correction_memory

        return correction_memory.stats()

    @app.post("/api/memory/clear", dependencies=[Depends(require_auth)])
    async def memory_clear() -> Dict[str, Any]:
        from veriscript.document import memory as correction_memory

        return {"removed": correction_memory.clear(),
                "enabled": correction_memory.memory_enabled()}

    # -- frontend ----------------------------------------------------------
    if os.path.isdir(FRONTEND_DIST):
        index = os.path.join(FRONTEND_DIST, "index.html")

        @app.get("/studio", include_in_schema=False)
        async def studio_page() -> FileResponse:
            return FileResponse(index)

        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="site")

    else:  # dev fallback: point the developer at the Vite dev server
        @app.get("/", include_in_schema=False)
        async def no_build() -> JSONResponse:
            return JSONResponse({
                "ok": True,
                "hint": "Frontend not built. Run `npm install && npm run build` in "
                        "webapp/frontend, or use the Vite dev server on :5173.",
            })

    return app
