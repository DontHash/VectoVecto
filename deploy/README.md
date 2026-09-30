# Deploying VeriScript cheaply (and safely)

Fixed-cost VPS + Caddy. One heavy worker, a hard daily budget, no
autoscaling and no metered surprises.

## Quick start

1. Any small VPS — 2 vCPU / 4 GB is comfortable; 1 vCPU / 2 GB works with
   slower first runs. Install Docker + the Compose plugin.
2. Point a domain/subdomain `A` record at the server IP.
3. On the server:

```bash
git clone https://github.com/DontHash/VeriScript
cd VeriScript/deploy
cp .env.example .env      # set SITE_ADDRESS (and ACME_EMAIL)
docker compose up -d --build
```

4. Open `https://<SITE_ADDRESS>` — TLS is automatic (Let's Encrypt).

## What protects you

| Layer | Guard | Where |
|---|---|---|
| Edge | TLS, request-body cap 13 MB, dial/read/write timeouts, hides `Server` | `deploy/Caddyfile` |
| App | per-IP rate limit: 6 runs / 2 min | `webapp/vvweb/security.py` |
| App | **hard daily budget: 300 runs/day (UTC), all clients** | `DailyQuota`, same file |
| App | 1 heavy worker; 25 s queue wait, then 503 | `webapp/vvweb/api.py` |
| App | per-run timeout 180 s | `webapp/vvweb/api.py` |
| App | uploads: 12 MB, 30 MP, magic bytes, decode check, 10 PDF pages | `security.py` / `pipeline.py` |
| App | results expire in 60 min; failed runs deleted | `storage.py` |
| App | optional HTTP Basic auth | `security.py` |
| Transport | uvicorn sheds connections beyond 32 | `webapp/server.py` |

The **daily budget is the billing kill-switch**: even a distributed flood
cannot make the demo process more than N pages per day. On a fixed-price VPS
the worst case is CPU contention — not a bill.

## Cost reality

| Option | Fixed cost | Notes |
|---|---|---|
| Small VPS (Hetzner / DO / Contabo / …) | ~$4–7/month | **recommended** — fixed bill, 2–4 GB RAM |
| Oracle Cloud Always Free (ARM) | $0 | if you can get capacity; 24 GB is overkill but fine |
| Hugging Face Spaces (free CPU) | $0 | sleeps when idle; enough RAM; needs `app_port: 8000` in the Space README metadata |
| Render free web service | $0 | 512 MB RAM — too small for the OCR models |
| Fly.io / Railway / metered hosts | varies | only with a hard usage cap + budget alert; pure pay-as-you-go is the risky option |

Using Cloudflare in front (free tier) adds DDoS/WAF protection; uncomment the
trusted-proxy block in `Caddyfile` so per-IP limits see the real visitor
instead of the Cloudflare edge.

## Tuning

All knobs are `VERISCRIPT_*` env vars (see [`../docs/DEPLOY.md`](../docs/DEPLOY.md)):

- `VERISCRIPT_WEB_DAILY_RUNS` — your spend ceiling (`0` disables the quota).
- `VERISCRIPT_WEB_USER` / `VERISCRIPT_WEB_PASSWORD` — make the demo private.
- `VERISCRIPT_WEB_FORWARDED_ALLOW_IPS` — set to your proxy address(es) when
  not using this Compose file.

## Backups

The only state is `webapp/runs` (ephemeral by design) and the model cache.
Nothing to back up.
