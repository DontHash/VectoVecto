# VeriScript Studio — web app

A clean web front door for the document pipeline: a short landing page
(hero, real examples, the offline/install story) with a **Try it** call to
action into `/studio`, where the full product runs on one page — drop a file,
watch the results, download the outputs.

- **Frontend** — SolidJS + TypeScript + Vite, hand-built design system
  (`src/styles/tokens.css`), three.js only for the studio's empty-state "desk"
  (lazy-loaded chunk; static fallback for reduced motion / no WebGL).
- **Type roles** — Instrument Serif for display, **Mukta for reading text**
  (the font the product itself embeds, so prose and Devanagari share one voice),
  IBM Plex Mono strictly for machine text (labels, filenames, tokens, numbers).
  The ink ladder is hard-coded, not alpha — 900 ≈ 17:1, 700 ≈ 9.3:1,
  500 ≈ 7.1:1 on the paper — so nothing washes out against the grain.
- **Backend** — FastAPI (`vvweb/`) around the repo's own pipeline
  (`veriscript.document.pipeline.run_document_pipeline`). The same pipeline the CLI runs;
  only the delivery is different.
- **Shared studio** — `lib/studio-state.ts` (one state factory) +
  `components/StudioPanel.tsx` / `StudioResults.tsx`, used by the `/studio`
  page (the landing page funnels there via **Try it**).

## Run it

```bash
# 1. backend deps (fastapi/uvicorn; the pipeline needs requirements.txt too)
pip install -r webapp/requirements-web.txt

# 2. build the frontend once
cd webapp/frontend && npm install && npm run build && cd ../..

# 3. serve (http://127.0.0.1:8000)
python webapp/server.py
```

Development: `cd webapp/frontend && npm run dev` (Vite on :5173, proxies `/api`
to :8000) while the backend runs on :8000.

Tools that regenerate web assets from the real pipeline output:

```bash
python webapp/tools/prepare_web_assets.py   # fonts + web-sized example images
python webapp/tools/export_queue_sample.py  # real review queue -> frontend data
```

## What is protected

| Surface | Guard |
|---|---|
| Uploads | extension + magic bytes, Pillow decode check, ≤ 12 MB, ≤ 30 MP, streamed to disk with a hard cap (headers are never trusted) |
| Abuse | per-IP sliding-window rate limit (default 6 runs / 2 min), one heavy worker, bounded queue (25 s) then 503 + `Retry-After` |
| Run time | hard timeout (180 s) per page; failed runs are deleted immediately |
| Data | runs live under `webapp/runs/<random-16-hex>/`; files are deleted after the retention window (default 60 min) by a periodic sweep; owner-only permissions (0700 dirs / 0600 manifests) on POSIX; nothing is logged about page content |
| Files | artifact names are a fixed whitelist; run ids are validated against `^[0-9a-f]{16}$`, so traversal is impossible |
| Proxy | real client IP only when `VERISCRIPT_WEB_FORWARDED_ALLOW_IPS` names the proxy (uvicorn proxy headers; legacy `VECTOVECTO_` name also accepted); otherwise the proxy IP is the rate-limit key |
| Cost | text-only gzip (≥ 1 KiB); hashed assets cached `immutable`, fonts/examples 1 week, HTML revalidates; no torch in the web image |
| Transport | security headers on every response (CSP without inline script/style, `nosniff`, `frame-ancestors 'none'`, referrer policy, COOP/CORP); API responses are `no-store` |
| Auth | optional HTTP Basic for `/api/*` — set `VERISCRIPT_WEB_USER` + `VERISCRIPT_WEB_PASSWORD` |

Configuration lives in `vvweb/config.py` (all env-driven). Nothing in the web
app can reach a private network path; the only outbound work is none.

## Honest notes

- **This demo runs the pipeline on the server.** The product's privacy guarantee
  ("no cloud calls, no telemetry") belongs to the local app; the web app says so
  on the page.
- Uploaded pages are processed once and deleted; the queue is in-memory and
  single-process, so a restart clears it.
- The invoice example in the studio is the original PNG, so the run reproduces
  the landing page's numbers exactly (27 tokens · 8 flagged · 4 conflicts). The
  letterpress example is a web-sized copy — a fresh run is real, but its flag
  counts can differ by one or two from the full-resolution run. OCR is sensitive
  to the bytes; the site never mixes a run's numbers across files.
- Only the document pipeline is exposed. The photo/upscale path is not (its
  weights are non-commercial — see docs/LICENSES.md); the web photo studio is
  parked with it (docs/PLAN_WEB_FULL.md, *Parked for later*).

## Deploy sketch

The server is a single uvicorn process; put it behind any reverse proxy that
terminates TLS and **forwards the real client IP** (so rate limiting works):

```bash
VERISCRIPT_WEB_HOST=0.0.0.0 \
VERISCRIPT_WEB_USER=studio \
VERISCRIPT_WEB_PASSWORD=… \
python webapp/server.py
```

For containers, the repo root `Dockerfile` already does this: it builds the
frontend in a Node stage, copies `dist/` next to `webapp/`, and runs
`python webapp/server.py` — no other moving parts.
