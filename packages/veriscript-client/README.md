# veriscript-client

Typed API client for a [VeriScript](https://github.com/DontHash/VeriScript)
server — the Python document pipeline behind the studio (`webapp/server.py`)
or the hosted demo. Zero runtime dependencies; Node >= 18 or any modern
browser.

The OCR itself runs in Python. This package is for integrations that drive a
VeriScript server: upload a page, inspect the review queue, apply
corrections, and build the privacy-first training export.

## Install

```bash
npm install veriscript-client
```

## Use

```ts
import { readFile } from "node:fs/promises";
import { createClient } from "veriscript-client";

const client = createClient({
  baseUrl: "https://veriscript.live", // omit for same-origin browsers
});

const page = await readFile("scan.png");
const run = await client.restore(new Blob([page], { type: "image/png" }), {
  filename: "scan.png",
  lang: "ne",          // "ne" | "hi" | "en"
});

console.log(run.review.length, "tokens to review");
for (const item of run.review) {
  console.log(item.index, item.text, item.flags, item.suggestions);
}

// Apply corrections (cumulative on the server; send only what is new).
const corrected = await client.correct(run.run_id, [
  { index: run.review[0]!.index!, bbox: run.review[0]!.bbox,
    original: run.review[0]!.text, corrected: "सही", action: "changed" },
]);
console.log(corrected.stats, corrected.files.corrected_txt);

// Download an artifact.
const txt = await client.downloadFile(run.run_id, "corrected.txt");
console.log(await txt.text());
```

### API

| Method | Endpoint | Notes |
|---|---|---|
| `health()` | `GET /api/health` | version, busy flag, quota limits |
| `restore(file, opts)` | `POST /api/restore` | upload a `Blob`/`File`; resolves when the run finishes |
| `correct(runId, corrections)` | `POST /api/runs/{id}/correct` | cumulative; returns updated queue + stats |
| `exportCorrections(runId, corrections, share)` | `POST /api/runs/{id}/corrections/export` | only `share` indices are packed — nothing leaves the machine otherwise |
| `memory()` / `clearMemory()` | `GET /api/memory`, `POST /api/memory/clear` | local correction memory (off by default on hosted servers) |
| `fileUrl(runId, name)` / `downloadFile(runId, name)` | `GET /api/runs/{id}/files/{name}` | artifacts are namespaced and expire with the run |

Errors surface as `VeriScriptError` with `status` and (for 429) `retryAfter`
seconds.

### Auth and limits

- Private servers (`VERISCRIPT_WEB_USER`/`VERISCRIPT_WEB_PASSWORD`) use HTTP
  Basic: `createClient({ auth: { username, password } })`.
- The hosted demo is rate- and quota-limited; `429` carries `Retry-After`.
  `health().limits` reports the current budgets.
- Runs expire after `ttl_minutes`; callers should keep their own copies.

## Development

```bash
npm install
npm run typecheck
npm test                                   # offline (fetch mocked)
VERISCRIPT_BASE_URL=http://127.0.0.1:8000 npm test   # + live smoke
npm run build
```

MIT, same as the parent project.
