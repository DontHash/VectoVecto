# Deploying the demo on Google Cloud Run

Serverless, scale-to-zero, one instance max. The bill is bounded by
`--max-instances 1` + the app's hard daily run and page budgets; a $5 budget
alert is part of the setup. Region used here: `us-central1`.

## Current deployment

| | |
|---|---|
| Service | `veriscript-demo` (us-central1, project `theproject-sr`) |
| URL | <https://veriscript.live> (also `www.veriscript.live`; the Cloud Run `…run.app` URL still serves) |
| DNS / certs | Namecheap DNS → Cloud Run domain mappings; Google-managed certificates provisioned 2026-10-01 |
| Image | `us-central1-docker.pkg.dev/theproject-sr/veriscript/app:latest` (sha-tagged per build) |
| Runtime | 1 vCPU / 2 GiB, concurrency 1, min 0 / max 1 instance, port 8000 |
| Env | `CLIENT_IP_MODE=xff-last`, `DAILY_RUNS=200`, `DAILY_PAGES=500`, `DAILY_PAGES_PER_CLIENT=20`, `RATE_MAX=6`, `RATE_WINDOW_S=120`, `MAX_CONNECTIONS=32`, `WARM_LANG=ne` |
| Guards verified | health reports the limits; a real page restored in ≈8–10 s warm / 25–45 s cold (2 vCPU measured *slower* — see Performance notes); the per-visitor page cap was verified live (the run over the limit returns 429 + `Retry-After` with the limit message) |

Update with: rebuild (`PRELOAD_MODELS=1`), push the same tag, then
`gcloud run deploy veriscript-demo --image <tag> --region us-central1 --project theproject-sr`.

## One-time setup (already done for `theproject-sr`)

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com --project theproject-sr
gcloud artifacts repositories create veriscript --repository-format=docker \
  --location=us-central1 --project=theproject-sr
gcloud billing budgets create --billing-account <BILLING_ACCOUNT> \
  --display-name "VeriScript demo - budget alert" --budget-amount 5USD \
  --threshold-rule percent=0.5 --threshold-rule percent=0.9 \
  --threshold-rule percent=1.0
```

## Build and push the image

Models are baked in (`PRELOAD_MODELS=1`) because Cloud Run's filesystem is
ephemeral — without it every cold start re-downloads the Devanagari model.

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build --build-arg PRELOAD_MODELS=1 \
  -t us-central1-docker.pkg.dev/theproject-sr/veriscript/app:v1 .
docker push us-central1-docker.pkg.dev/theproject-sr/veriscript/app:v1
```

## Deploy

```bash
gcloud run deploy veriscript-demo \
  --image us-central1-docker.pkg.dev/theproject-sr/veriscript/app:v1 \
  --region us-central1 --project theproject-sr \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 1 --memory 2Gi --concurrency 1 \
  --min-instances 0 --max-instances 1 \
  --timeout 300 \
  --set-env-vars "VERISCRIPT_WEB_HOST=0.0.0.0,\
VERISCRIPT_WEB_CLIENT_IP_MODE=xff-last,\
VERISCRIPT_WEB_DAILY_RUNS=200,\
VERISCRIPT_WEB_DAILY_PAGES=500,\
VERISCRIPT_WEB_DAILY_PAGES_PER_CLIENT=20,\
VERISCRIPT_WEB_RATE_MAX=6,VERISCRIPT_WEB_RATE_WINDOW_S=120,\
VERISCRIPT_WEB_MAX_CONNECTIONS=32,\
VERISCRIPT_WEB_WARM_LANG=ne"
```

Notes:

- `CLIENT_IP_MODE=xff-last` is the spoof-safe setting for Cloud Run: the
  Google front end appends the real client IP to `X-Forwarded-For`, so the
  rate limiter keys on the visitor, not a shared edge IP.
- `--max-instances 1` + `--concurrency 1` is the same single-heavy-worker
  profile as the local app; a flood queues and then sheds.
- `--min-instances 0` keeps idle cost at zero; the first request after a
  while pays a cold start (models are on the image, so it is short).
- For a private demo, drop `--allow-unauthenticated` (Cloud Run IAM) **or**
  set `VERISCRIPT_WEB_USER` / `VERISCRIPT_WEB_PASSWORD`.

## Performance notes (measured on the live service)

Single invoice page (`invoice-hero.jpg`), warm instance:

| Config | Warm total | Processing |
|---|---|---|
| **1 vCPU, `OMP_NUM_THREADS=2` (image default) — current** | ≈8–10 s | ≈8–9 s |
| 2 vCPU, `OMP_NUM_THREADS=2` | 12.7–13.5 s | ≈11.9 s |
| 1 vCPU, `OMP_NUM_THREADS=1` | 9.9–10.2 s | ≈9.2 s |

Counter-intuitive but measured: **more vCPU is slower** for this workload —
the page is largely single-stream ONNX inference, so extra threads contend
instead of parallelising. Cold start (first request after idle) is ~25–45 s
regardless, because the OCR models load and warm then. Keep 1 vCPU and the
image's `OMP_NUM_THREADS=2`; if you change either, re-measure with a few warm
runs (the rate limit allows 6 / 2 min).

## Auto-deploy (GitHub Actions)

Every push to `main` that passes CI builds the image and deploys a new
revision — `.github/workflows/deploy.yml`. Manual runs: Actions → *Deploy demo
(Cloud Run)* → Run workflow. Auth is keyless (Workload Identity Federation);
one-time setup, already done for this project:

```bash
# one-time project APIs (iamcredentials powers WIF token minting)
gcloud services enable iamcredentials.googleapis.com --project=theproject-sr

# deployer service account
gcloud iam service-accounts create veriscript-deployer --project theproject-sr
SA=veriscript-deployer@theproject-sr.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding theproject-sr \
  --member="serviceAccount:$SA" --role=roles/run.developer
gcloud projects add-iam-policy-binding theproject-sr \
  --member="serviceAccount:$SA" --role=roles/artifactregistry.writer
gcloud iam service-accounts add-iam-policy-binding \
  1066484194377-compute@developer.gserviceaccount.com --project=theproject-sr \
  --member="serviceAccount:$SA" --role=roles/iam.serviceAccountUser

# GitHub OIDC pool, restricted to this repository
gcloud iam workload-identity-pools create github --project=theproject-sr --location=global
gcloud iam workload-identity-pools providers create-oidc veriscript \
  --project=theproject-sr --location=global --workload-identity-pool=github \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='DontHash/VeriScript'"
gcloud iam service-accounts add-iam-policy-binding $SA --project=theproject-sr \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/1066484194377/locations/global/workloadIdentityPools/github/attribute.repository/DontHash/VeriScript"
```

The workflow tags each image with the commit SHA and `latest`, deploys the SHA
tag, then polls `/api/health` until the new revision answers.

## Update workflow

1. Change code, run the suite: `python -m pytest tests/ -q`.
2. Build with a **new tag** (keeps rollback trivial) and push:
   `docker build --build-arg PRELOAD_MODELS=1 -t …/app:v2 . && docker push …/app:v2`.
3. Deploy the new image — every other setting (env vars, CPU, limits,
   concurrency) persists on the service:
   `gcloud run deploy veriscript-demo --image …/app:v2 --region us-central1 --project theproject-sr`.
4. Verify: `curl $URL/api/health` and one warm `/api/restore`.
5. Roll back: `gcloud run revisions list --service veriscript-demo --region us-central1
   --project theproject-sr`, then
   `gcloud run services update-traffic veriscript-demo --to-revisions <REV>=100
   --region us-central1 --project theproject-sr`.
   Revisions pin their image digest, so rollback works even if a tag was
   overwritten; revisions themselves cost nothing.

## Verify

```bash
URL=$(gcloud run services describe veriscript-demo --region us-central1 \
  --project theproject-sr --format "value(status.url)")
curl "$URL/api/health"
curl -F "file=@webapp/frontend/public/examples/invoice-hero.jpg" \
     -F "lang=en" "$URL/api/restore"
```

## Cost expectations

- **Cloud Run**: free tier covers 180k vCPU-s + 360k GiB-s + 2M requests per
  month. At the daily budget (200 runs / 500 pages) with typical 5–15 s pages
  this stays inside the free tier; `--max-instances 1` bounds the worst case
  even if someone floods the demo.
- **Artifact Registry**: ~$0.10/GB/month beyond the 0.5 GB free tier; the
  image is 442 MB, so storage sits inside the free tier (~$0).
- A $5 budget alert notifies at 50/90/100% of $5.

## Logs

```bash
gcloud run services logs read veriscript-demo --region us-central1 \
  --project theproject-sr --limit 50
```

## Monitoring & alerts

Set up for this service (2026-09-30):

- **Uptime check** `VeriScript demo /api/health` — HTTPS GET every 5 min from
  USA / Europe / Asia-Pacific, 2xx expected, 10 s timeout.
- **Alert policy** `VeriScript demo is down` — emails the account address when
  any region reports a failed probe for 5 minutes (auto-closes 30 min after
  recovery).
- **Budget alerts** — `VeriScript demo - budget alert` ($5; 50/90/100%) and
  the project-wide `theProject-SR cap 100usd`.

Inspect or change:

```bash
gcloud monitoring uptime list-configs --project=theproject-sr
gcloud monitoring policies list --project=theproject-sr
```

## Custom domain

`veriscript.live` is registered at Namecheap (2026-09-30). The order matters:

1. **Register the domain** at any registrar (≈$10–12/year).
2. **Verify ownership** with Google: in the Cloud Console open *Cloud Run →
   veriscript-demo → Manage custom domains* (the flow creates the Search
   Console TXT record for you), or add the TXT record manually in
   [Search Console](https://search.google.com/search-console).
3. **Create the mapping.** Fully managed Cloud Run needs the **beta** command
   — the GA `gcloud run domain-mappings create` is the Anthos (Knative)
   variant and rejects `--region`:

   ```bash
   gcloud components install beta   # needs an elevated shell: the SDK lives in Program Files
   gcloud beta run domain-mappings create --service veriscript-demo \
     --domain veriscript.live --region us-central1 --project theproject-sr
   ```

   Or do step 3 from the Console UI (*Manage custom domains*), which skips the
   beta install entirely.
4. **DNS**: add the records the command returns — A/AAAA for the apex,
   CNAME to `ghs.googlehosted.com` for subdomains.

Done for this service (2026-09-30): both `veriscript.live` (4×A + 4×AAAA) and
`www.veriscript.live` (CNAME) are mapped; the mappings were created through
the Cloud Run Admin API, so no `beta` component was needed.

Domain mapping is free and fine for a demo. For a production setup, prefer a
global external Application Load Balancer + serverless NEG (health checks,
CDN, managed certificates).

Cloudflare in front is possible but needs care: with Cloudflare proxying to
Cloud Run, Google's front end appends the *Cloudflare edge* IP to
`X-Forwarded-For`, so the app's `xff-last` client-IP mode would key the rate
limit and the per-visitor quota on the Cloudflare edge — not the visitor. Only
add Cloudflare if you also add a `CF-Connecting-IP` client-IP mode to
`webapp/vvweb/security.py`.
