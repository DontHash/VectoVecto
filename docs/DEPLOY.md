# Deploying the web app

The product is local-first: `python webapp/server.py` binds to `127.0.0.1`
and never calls out. Hosting it is optional and only needed to demo without
shipping anything to the browser user. Nothing in this repository contains
training data or checkpoints — a hosted instance serves the pipeline, the OCR
models that come with the `rapidocr` wheel, and nothing else.

## Docker (any host)

```bash
docker build -t vectovecto .
docker run --rm -p 8000:8000 vectovecto
# → http://127.0.0.1:8000
```

The image builds the SolidJS studio in a Node stage and serves it from the
FastAPI backend. Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `VECTOVECTO_WEB_HOST` | `0.0.0.0` (image) | bind address |
| `VECTOVECTO_WEB_PORT` | `8000` | port (also the healthcheck) |
| `VECTOVECTO_WEB_WARM_LANG` | `ne` | OCR language warmed at startup (covers the one-time Devanagari model download; set `en` to warm Latin only) |
| `VECTOVECTO_WEB_USER` / `VECTOVECTO_WEB_PASSWORD` | unset | when both are set, `/api/*` requires HTTP basic auth |
| `VECTOVECTO_WEB_MAX_UPLOAD_MB` | `12` | upload cap per page |
| `VECTOVECTO_WEB_MAX_MP` | `30` | image megapixel cap |
| `VECTOVECTO_WEB_TIMEOUT_S` | `180` | per-page processing timeout |
| `VECTOVECTO_WEB_CONCURRENCY` / `VECTOVECTO_WEB_QUEUE_WAIT_S` | `1` / `25` | single heavy worker; queue beyond it, then 503 |
| `VECTOVECTO_WEB_TTL_MINUTES` | `60` | artifact retention before the periodic sweep |
| `VECTOVECTO_WEB_RATE_MAX` / `VECTOVECTO_WEB_RATE_WINDOW_S` | `6` / `120` | per-IP sliding window |
| `VECTOVECTO_LEXICON` | `data/lexicon/nepali_lexicon_v1.txt` | optional `unknown_word` lexicon; not in the image (see `scripts/fetch_nepali_lexicon.py`) |
| `VECTOVECTO_LOG_LEVEL` | `INFO` | stderr log level (`DEBUG`, `WARNING`, ...) |

Full list and defaults: [`webapp/vvweb/config.py`](../webapp/vvweb/config.py).

For a private demo, always set the auth pair:

```bash
docker run --rm -p 8000:8000 \
  -e VECTOVECTO_WEB_USER=demo -e VECTOVECTO_WEB_PASSWORD='change-me' \
  vectovecto
```

The image runs as a non-root user, excludes `data/`, `weights/`, `artifacts/`
and the frontend build state (see `.dockerignore`), and carries a healthcheck
on `/api/health`.

## Managed platforms

- **Hugging Face Spaces (Docker SDK):** create a Space, push this repo, set the
  Space to use the existing `Dockerfile` (port `8000`), add the auth secrets in
  the Space settings. The free tier is CPU-only, which matches the image.
- **Render / Railway / Fly.io:** point the service at the `Dockerfile`, expose
  port `8000` (or set `VECTOVECTO_WEB_PORT` to the one the platform expects),
  add the same environment variables. Give the first request ~60–90 s of
  start-up budget (model warm-up).
- **A VPS:** `docker compose` or plain `docker run` behind a TLS reverse proxy
  (Caddy/nginx). Keep basic auth on unless the instance is private. The proxy
  must forward the real client IP or the rate limit collapses onto one key.

## Resource profile

| Resource | Value |
|---|---|
| Image size | ~1.2 GB (CPU torch + OCR models + Node build stage is discarded) |
| RAM | 2 GB minimum, 4 GB comfortable |
| Cold start | ~40–90 s (torch + ONNX warm-up) |
| Throughput | ~1 s/page for clean scans; letterpress ~2–4 s/page with the digit re-pass |

## Not in the web studio

The digit verifier (`--digit-verifier bodhan`), the Devanagari line reader
(`--deva-lines`) and the mixed-page router stay opt-in CLI features. The
photo upscaler is CLI-only (its weights are non-commercial) and its planned
web studio is parked — see [PLAN_WEB_FULL.md](PLAN_WEB_FULL.md).

## What is deliberately not hosted

- The optional bodhan digit verifier (~1.9 GB, Indic Open Model License — no
  third-party hosting) stays off unless you accept its license on your own
  infrastructure.
- The photo upscaler weights (CC-BY-NC-SA) are excluded from the image.
- No evaluation corpora, no training sets, no checkpoints.
