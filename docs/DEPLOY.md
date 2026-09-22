# Deploying the web app

The product is local-first: `python app.py` binds to `127.0.0.1` and never
calls out. Hosting it is optional and only needed to demo without shipping the
desktop build. Nothing in this repository contains training data or
checkpoints — a hosted instance serves the pipeline, the OCR models that come
with the `rapidocr` wheel, and nothing else.

## Docker (any host)

```bash
docker build -t vectovecto .
docker run --rm -p 7860:7860 vectovecto
# → http://127.0.0.1:7860
```

Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `VECTOVECTO_HOST` | `0.0.0.0` (image) | bind address |
| `VECTOVECTO_PORT` | `7860` | port |
| `VECTOVECTO_USER` / `VECTOVECTO_PASSWORD` | unset | when both are set, Gradio requires HTTP basic auth |
| `VECTOVECTO_DOCUMENT_ONLY` | unset | `1` hides the photo tab (its upscale weights are non-commercial and not in the image) |

For a private demo, always set the auth pair:

```bash
docker run --rm -p 7860:7860 \
  -e VECTOVECTO_USER=demo -e VECTOVECTO_PASSWORD='change-me' \
  -e VECTOVECTO_DOCUMENT_ONLY=1 \
  vectovecto
```

The image runs as a non-root user, excludes `data/`, `weights/`, `artifacts/`,
`legacy/`, `evals/`, `scripts/` and `deva_crnn/` (see `.dockerignore`), and
carries a healthcheck on the Gradio root URL.

## Managed platforms

- **Hugging Face Spaces (Docker SDK):** create a Space, push this repo, set the
  Space to use the existing `Dockerfile`, add `VECTOVECTO_DOCUMENT_ONLY=1` and
  the auth secrets in the Space settings. The free tier is CPU-only, which
  matches the image.
- **Render / Railway / Fly.io:** point the service at the `Dockerfile`, expose
  port `7860`, add the same environment variables. Give the first request
  ~60–90 s of start-up budget (model warm-up).
- **A VPS:** `docker compose` or plain `docker run` behind a TLS reverse proxy
  (Caddy/nginx). Keep basic auth on unless the instance is private.

## Resource profile

| Resource | Value |
|---|---|
| Image size | ~1.1 GB (CPU torch + OCR models) |
| RAM | 2 GB minimum, 4 GB comfortable |
| Cold start | ~40–90 s (torch + ONNX warm-up) |
| Throughput | ~1 s/page for clean scans; letterpress ~2–4 s/page with the digit re-pass |

## Devanagari line reader (optional, server-side)

The trained recognizer weights are **not in this repository**. To enable the
opt-in reader in a hosted instance:

```bash
docker run --rm -p 7860:7860 \
  -v /srv/vectovecto/weights:/app/weights:ro \
  -e VECTOVECTO_DEVA_CKPT=/app/weights/deva_crnn_h48w512.pt \
  vectovecto
```

Without the mount the app runs exactly as before (the reader is off by
default; `--deva-lines on` is the opt-in). Training recipe and measured
scope: [TRAINING.md](TRAINING.md).

## What is deliberately not hosted

- The optional bodhan digit verifier (~1.9 GB, Indic Open Model License — no
  third-party hosting) stays off unless you accept its license on your own
  infrastructure.
- The photo upscaler weights (CC-BY-NC-SA) are excluded from the image.
- No evaluation corpora, no training sets, no checkpoints.
