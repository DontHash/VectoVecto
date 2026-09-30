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

## Cloudflare in front (free tier)

Standard practice for a public demo on a VPS: hides the origin IP, absorbs
L3/L4 and basic L7 DDoS, terminates TLS at the edge and adds WAF/bot rules —
without changing the app.

Setup:

1. Add the domain to Cloudflare, point the registrar's nameservers at it, and
   create a proxied (orange-cloud) `A` record to the server IP.
2. In `Caddyfile`, uncomment the `trusted_proxies` + `client_ip_headers`
   block and paste the current ranges from <https://www.cloudflare.com/ips/>
   — without it the app rate limit keys the Cloudflare edge IP, not the
   visitor.
3. SSL/TLS mode: **Full (strict)** once Caddy has its certificate. On the
   very first deploy, leave DNS-only until Caddy got the cert, then switch
   the record to proxied.
4. Cache rule: **bypass** `https://<your-domain>/api/*` (the app already
   sends `no-store`; this is belt-and-braces).
5. Optional: enable Bot Fight Mode.

What it does **not** replace:

- The app's own guards stay primary. Free-plan rate-limiting rules are
  limited, so the per-IP window + the daily budget are still the real limits.
- The free plan cuts origin responses at ~100 s (error 524). Our worst case
  is a 25 s queue plus a 180 s page timeout, so for a Cloudflare-fronted demo
  set `VERISCRIPT_WEB_TIMEOUT_S: "95"` in `docker-compose.yml` and keep PDF
  runs short — otherwise a very long run gets a Cloudflare error while the
  origin is still finishing.

If you host on Google Cloud Run instead, this layer is optional: Google's
edge already provides TLS and baseline DDoS protection; Cloudflare would only
add WAF/bot rules and a custom domain setup.

## Tuning

All knobs are `VERISCRIPT_*` env vars (see [`../docs/DEPLOY.md`](../docs/DEPLOY.md)):

- `VERISCRIPT_WEB_DAILY_RUNS` — your spend ceiling (`0` disables the quota).
- `VERISCRIPT_WEB_USER` / `VERISCRIPT_WEB_PASSWORD` — make the demo private.
- `VERISCRIPT_WEB_FORWARDED_ALLOW_IPS` — set to your proxy address(es) when
  not using this Compose file.

## Backups

The only state is `webapp/runs` (ephemeral by design) and the model cache.
Nothing to back up.
