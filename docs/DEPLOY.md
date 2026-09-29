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
| `VECTOVECTO_WEB_FORWARDED_ALLOW_IPS` | unset | enable uvicorn proxy headers (real client IP for rate limiting) only behind a trusted reverse proxy; e.g. `127.0.0.1`, a subnet, or `*` |
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

## Hosting hardening & cost checklist

What the app already does:

- **Input**: upload limits (12 MB, 30 MP, 10 PDF pages), magic-byte + Pillow
  decode validation, streamed uploads with a hard cap (headers never trusted).
- **Abuse**: per-IP sliding-window limit (429 + `Retry-After`), one heavy
  worker with a bounded queue (503 + `Retry-After`), per-run timeout; the
  limiter table is capped at 20k keys so an IP spray cannot grow memory.
- **Data**: runs under `webapp/runs/<16-hex>/`, deleted by TTL (default
  60 min); failed runs deleted immediately; owner-only permissions (0700
  dirs / 0600 manifests) on POSIX; nothing is logged about page content.
- **Transport**: security headers on every response (CSP, nosniff,
  `frame-ancestors 'none'`, COOP/CORP, no-referrer), `Cache-Control: no-store`
  on `/api/*`.
- **Cost**: text-only gzip (≥ 1 KiB, never re-compresses PNG/PDF); hashed
  assets are `immutable`, fonts/examples cache one week, HTML revalidates;
  no torch in the web image; models warm up once at start.

What the deployment should add:

- TLS with HSTS at the reverse proxy; **forward the real client IP** and set
  `VECTOVECTO_WEB_FORWARDED_ALLOW_IPS` (rate limiting depends on it).
- Set the `VECTOVECTO_WEB_USER`/`PASSWORD` pair for anything public; don't
  cache `/api/*` at the proxy (the app already sends `no-store`).
- Keep one uvicorn process: the limiter, queue and run store are per-process.
  Scale out only with per-IP limits at the edge (or a shared limiter later).
- Optional: run the container read-only with a tmpfs for `webapp/runs`
  (`--read-only --tmpfs /app/webapp/runs`), set memory limits, and alert on
  429/503 rates.

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
