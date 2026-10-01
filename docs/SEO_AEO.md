# Discoverability plan — SEO, AEO and distribution

Research date: 2026-10-01. Audited: `veriscript.live`, GitHub
`DontHash/VeriScript`, PyPI, and the competitor SERPs for the
Nepali/Devanagari OCR query space. External claims are sourced at the end;
industry claims are marked as such.

## 0. Diagnosis

The product is real and differentiated; the **discovery surface is empty**.
Nothing can rank, and nothing can be cited, that does not exist.

| Surface | State (2026-10-01) | Consequence |
|---|---|---|
| veriscript.live | SPA. Initial HTML = `<title>` + meta description + `<div id="root">`; all copy is JS-rendered | Non-rendering crawlers see an empty page; the best copy on the site is invisible |
| URL count | 2 routes (`/`, `/studio`) | no page exists that could rank for any query |
| `robots.txt` | **404** | no sitemap pointer, no explicit AI-crawler policy |
| `sitemap.xml` | **404** | no discovery aid |
| Structured data | none | no SoftwareApplication / FAQ / HowTo / Dataset signals |
| `llms.txt` | none | low impact either way (§5) |
| GitHub | 0 stars / 0 forks / no releases; strong README; topics set | no social proof, no distribution |
| PyPI | **not published** | no trusted indexable project page; install is `git clone` only |
| Third-party mentions | none found | the #1 predictor of AI visibility is missing entirely (§2) |
| Competitor SERP | FastOCR, OLOCR, Toolghar, personal tool pages rank for "Nepali OCR" with thin pages and unreferenced claims ("96% accuracy") | the query space we should own is held by pages with no evidence behind them |

**One-line diagnosis: Phase 0 is surface, not content strategy.** The
evidence already in this repo (frozen sets, CIs, head-to-head tables) is
better content than anything currently ranking — it just is not on the web
in a crawlable form.

## 1. Who we are selling to (query themes)

| Segment | What they search | Page that should exist |
|---|---|---|
| Nepali/Hindi end users | `nepali ocr`, `photo to text nepali`, `searchable pdf nepali`, `citizenship card text extract`, `lalpurja digitize` | `/nepali-ocr`, `/hindi-ocr`, `/searchable-pdf` |
| Privacy / compliance buyers | `offline ocr software`, `ocr without cloud`, `private ocr`, `self-hosted ocr`, `abbyy alternative offline` | `/offline-ocr`, `/compare/abbyy` |
| Accuracy-sensitive professionals | `ocr that flags errors`, `ocr verification`, `bill digitization accuracy`, `court register digitization` | `/why-review-queue`, `/benchmark` |
| Developers | `devanagari ocr python`, `pdf to markdown open source`, `searchable pdf python` | `/pdf-to-markdown`, GitHub + PyPI |
| AI-engine questions | `best nepali ocr`, `is nepali ocr accurate`, `free nepali ocr`, `offline alternatives to abbyy` | answer-first pages + third-party mentions (§5) |

**Wedge:** no competitor in this query space ships measured evidence. Every
VeriScript page can carry numbers that link to frozen manifests — the
"96% accuracy" pages cannot. The honesty feature is the content strategy:
*"it will not invent the numbers on your bill"* is the hook, the benchmark
is the proof.

## 2. How discovery works in 2026 (the rules we play by)

Sourced facts, not opinions:

1. **AI answer engines are now a real channel.** ~12–18% of English
   informational queries go to AI engines as of Q1 2026 (was <2% a year
   earlier); AI-referred traffic converts at higher rates [1]. ChatGPT alone
   reports >900M weekly users [2].
2. **Question and comparison shapes are the ones AI answers.** Comparison
   queries trigger a Google AI Overview ~95% of the time; question-format
   ~86% [3].
3. **Off-site mentions beat everything on-site.** The strongest predictor of
   AI visibility is how often your brand is mentioned across *other*
   websites — ~3× more predictive than backlinks; ~85% of LLM citations for
   category queries come from third-party sources, and you are ~6.5× more
   likely to be cited through someone else's page than your own [2][4]. Top
   quartile for web mentions earns ~10× more AI Overview appearances — a
   cliff, not a curve [4].
4. **AI crawlers mostly do not run JavaScript** (GPTBot and several others
   fetch the initial HTML only); Google AI Overviews/Mode use the normal
   search index [1][5]. Content that only exists after hydration is at risk
   on both fronts.
5. **Structured data still helps extraction — but pick types that still
   pay.** FAQ rich results were retired 2026-05-07 (HowTo rich results in
   2023), so neither is a visible-SERP play any more; keep FAQ/HowTo blocks
   as harmless semantics for AI engines, not as a rich-result bet. The types
   that still carry weight: `SoftwareApplication`, `Organization`,
   `Dataset`, `Article`, `BreadcrumbList` [5][7].
6. **Freshness matters** for whether citations stick [1].
7. **`llms.txt` is not a lever.** Google ignores it by policy; log studies
   show almost no AI citation bot fetches it (408 of 515M+ AI-bot visits in
   one 90-day study; 97% of files zero requests in May 2026) [6]. Ship it as
   cheap forward-compatibility, expect nothing from it.

## 3. Phase 0 — make the site crawlable and citable (1–3 days)

Everything below is small, concrete, and in this repo's control.

**3.1 `robots.txt` + `sitemap.xml`** — add as static files in
`webapp/frontend/public/` (Vite copies `public/` to `dist/`; the FastAPI
`StaticFiles(html=True)` mount serves them at the root). `robots.txt` should
point to the sitemap and explicitly allow the retrieval crawlers
(`OAI-SearchBot`, `PerplexityBot`, `Claude-Web`/`ClaudeBot`, `Google-Extended`,
`Bingbot`). **If Cloudflare fronts the domain, check the zone's AI-bot /
"block AI training" toggles — the default posture may already be blocking
these.**

**3.2 Pre-rendered content pages.** Recommendation: hand-written static HTML
in `webapp/frontend/public/<slug>/index.html` (Option A) — zero framework
change, served today by the existing `html=True` mount; add explicit FastAPI
`FileResponse` routes for the clean slugs if trailing-slash redirects are
awkward. Options B/C (SolidStart SSR, separate Astro docs sub-site) are more
capable but not needed until page count grows.

**3.3 Home page.** Prerender the landing copy into `index.html` (post-build
inject or a Vite plugin) or at minimum mirror the key copy in the static
HTML. Add: canonical, OG/Twitter tags, and JSON-LD
(`SoftwareApplication` + `Organization` + `WebSite`).

**3.4 Per-page SEO/AEO template** (see §4 rules): unique title, meta
description, canonical, OG image, and schema (`FAQPage`, `HowTo`,
`Dataset`, `BreadcrumbList` where they fit).

**3.5 Indexing + measurement plumbing.** Google Search Console, Bing
Webmaster Tools (Bing feeds Copilot and ChatGPT search), IndexNow ping on
deploy, and referrer tracking that separates AI engines
(`chatgpt.com`, `perplexity.ai`, `gemini.google.com`,
`copilot.microsoft.com`, `claude.ai`).

**3.6 Distribution hygiene.** Publish to PyPI (indexable, trusted project
page; `pip install veriscript` beats `git clone`), cut a GitHub release with
badges, add an OG image to the repo social preview.

**3.7 `llms.txt`** — one hour, hygiene only. Curated links to the benchmark,
tool pages, docs. No expectations (§2.7).

**Status 2026-10-01 — shipped (zero-credential items):** `3.1` robots.txt +
sitemap.xml (AI crawlers explicitly allowed), `3.2` first static content page
(`/nepali-ocr/`, answer-first, FAQPage/Breadcrumb JSON-LD), `3.3` home
prerender fallback + OG/Twitter/canonical + `SoftwareApplication` JSON-LD,
`3.6` social card (`scripts/make_og_image.py`), `3.7` llms.txt. Remaining:
`3.5` GSC/Bing Webmaster + IndexNow accounts, `3.6` PyPI publish + GitHub
release, and the launch/marketing half (§5–6). DataForSEO deferred — see §10.

## 4. Phase 1 — own the query space (1–2 weeks)

Static pages, every one with the demo one click away. The material already
exists in `README.md`, `EVALUATION.md`, `BENCHMARK.md` and `MARKET.md` —
this is repackaging, not new research.

| URL | Target queries | Content core |
|---|---|---|
| `/nepali-ocr` | nepali ocr, nepali image to text | tool + use cases (citizenship, lalpurja, certificates), honest accuracy numbers, review-queue demo, FAQ |
| `/hindi-ocr` | hindi ocr, devanagari ocr | same, Hindi framing |
| `/offline-ocr` | offline ocr, ocr without cloud, private ocr | air-gap story, no-account/no-telemetry, install + Docker, cost comparison vs cloud read-tier |
| `/searchable-pdf` | searchable pdf nepali/hindi, make scanned pdf searchable | PDF in → searchable PDF + review queue, bundled Mukta font, CLI + studio |
| `/pdf-to-markdown` | pdf to markdown, devanagari ocr python | developer framing, JSON/queue, link to PyPI + GitHub |
| `/benchmark` | devanagari ocr benchmark, nepali ocr accuracy | canonical head-to-head (CER + CIs, methodology, reproduction commands, raw JSON links), `Dataset` schema |
| `/compare/tesseract` etc. | tesseract devanagari, abbyy fine reader alternative, ocrmypdf alternative | comparison tables — and say where we lose (handwriting, scale, brand) |
| `/why-review-queue` | ocr errors, ocr hallucination, digit verification | the honesty story with measured recall/ECE numbers |

**Rules for every page** (this is what AI engines extract):

1. Answer-first: 40–60 words under the H1 that survives being lifted out of
   context.
2. H2s phrased as the questions people ask.
3. Comparison tables and FAQ blocks wherever the topic allows.
4. Measured numbers only, each linking to its frozen set / raw JSON;
   "updated" date visible.
5. Nepali (and where it earns it, Hindi) versions of the tool pages with
   `hreflang` — the audience searches in both scripts; competitors barely do
   this.

## 5. Phase 2 — become the citable third-party source (ongoing)

§2.3 is the whole point: AI visibility is won off-site. On-site pages give
crawlers something to cite; these actions make *other* pages cite us.

1. **Benchmark as the flagship artifact.** Publish the head-to-head as a
   citable page + DOI (Zenodo) + the raw JSON. The 2026 arXiv stress-test
   ([2606.29213](https://arxiv.org/abs/2606.29213)) shows researchers are
   asking exactly this question; nobody publishes page-level Devanagari CER
   with CIs *plus* alternative readings. Contact the author, cite the paper,
   offer the harness — benchmarks get cited, and citations compound.
2. **"State of Devanagari OCR 2026"** report (the bake-off: Surya, Qwen,
   PaddleOCR, RapidOCR, Tesseract, VeriScript). Tables and statistics are
   what AI engines lift; this is also the kind of post HN/Reddit actually
   reads.
3. **Directories and lists** (justified PRs, honest entries):
   `awesome-ocr`, self-hosted/privacy lists, alternativeto.net (alternative
   to ABBYY / OCRmyPDF / Tesseract), `selfh.st`, Product Hunt, DevHunt.
4. **Communities, with the evidence, not the pitch:**
   - Show HN / r/selfhosted / r/LocalLLaMA (local-first + measured numbers)
   - r/Nepal, r/NepaliLanguage, Nepal tech groups (the actual users)
   - Stack Overflow answers where the question is real (searchable-PDF
     Python, Devanagari OCR pipelines)
   - **Wikisource / Wikimedia Indic proofreading** — a real, ongoing need
     for better Devanagari OCR; Wikimedia pages are heavily cited by AI
     engines
   - Libraries/archives: TU/KU libraries, e-Granthalaya users, Nepal
     government digitization programmes; `ailiteracynepal.com`'s Devanagari
     OCR course page is a natural listing
5. **PyPI + Hugging Face presence** — trusted domains that both search and
   AI engines read.

## 6. Phase 3 — launch moments, sequenced

Don't shotgun. Sequence so the spike lands on a site with substance (§4
done first), because the spike's mentions/backlinks are what feed AI
visibility later.

1. **Show HN**: *"VeriScript – offline OCR that flags uncertain digits
   instead of inventing them"*. Needs: one-command quickstart, 90-second
   demo GIF, README polish (already strong), a maintainer in the comments
   with the measured numbers ready.
2. **Product Hunt / DevHunt** in the same window; prepare assets and a
   hunter.
3. Reddit cross-posts, framed per community (evidence for technical subs,
   use cases for Nepal subs).
4. **Desktop installer** (packaging spec already exists) + `winget`/store
   listing — distribution is discovery: installers rank in stores and get
   reviewed.

## 7. Measurement (monthly, ~1 hour)

**Baseline prompt set** — run the same ~20 queries across ChatGPT,
Perplexity, Google AIO/Mode, Gemini, Copilot; record: answer appears? we
cited? who is? Queries: `best nepali ocr`, `nepali ocr accuracy`, `offline
ocr software`, `make scanned nepali pdf searchable`, `devanagari ocr open
source`, `ocr that detects errors`, `abbey alternative offline`, `pdf to
markdown open source`, plus the Hindi variants.

**Dashboards:** GSC (impressions/clicks/queries/pages), Bing WMT, IndexNow
submissions, referrers (AI engines, HN, Reddit), GitHub traffic + stars,
PyPI downloads.

**90-day targets (first pass, honest):**

| Metric | Target |
|---|---|
| Indexable pages | ≥ 10 (Phase 1 shipped) |
| "nepali ocr"-adjacent query | top-20 for at least one |
| AI citation | cited once on the prompt set |
| Third-party mentions | ≥ 3 (list/community/review) |
| GitHub stars | meaningful launch bump (Show HN range, not a number to promise) |
| PyPI | live, installable |

## 8. What not to do

- **Do not block AI crawlers.** Watch Cloudflare defaults (AI-bot toggle,
  "block AI training") — a one-click default can erase the whole AEO plan.
- **Do not sell `llms.txt` as the strategy** — no measured citation lift,
  Google ignores it (§2.7).
- **Do not copy the competitor tool-page formula.** Thin pages with
  unreferenced claims are why they can be outranked; ours must stay
  evidence-rich.
- **Do not overclaim.** Every number links to a frozen manifest. One
  invented metric destroys the honesty brand — that brand is the moat.
- **Do not chase generic English "OCR" head terms** (ABBYY/Adobe/Google
  budgets). Own Devanagari, privacy, and honesty long-tail.
- **Do not buy links or guest-post farms.** It contradicts the brand and
  the engines that matter here penalize it.

## 9. Priority

| # | Action | Effort | Impact | Depends on |
|---|---|---|---|---|
| 1 | `robots.txt`, sitemap, IndexNow, GSC/Bing, Cloudflare check | hours | foundational | — |
| 2 | Prerender home + OG/canonical/JSON-LD | 1 day | high | — |
| 3 | `/nepali-ocr` + `/hindi-ocr` pages (answer-first, FAQ schema) | 1–2 days | high | 2 |
| 4 | `/benchmark` canonical page | 1–2 days | high | 2 |
| 5 | PyPI publish + GitHub release | hours | high (trust + channel) | — |
| 6 | Show HN + PH/DevHunt + Reddit | 1 week prep | high (mentions → AEO) | 3–5 |
| 7 | awesome-list PRs, alternativeto, selfh.st | 2–3 days | medium | 3 |
| 8 | "State of Devanagari OCR" report + arXiv author contact | 1 week | high (citations) | 4 |
| 9 | Comparison pages | 2–3 days | medium | 3 |
| 10 | AI-citation baseline + monthly check | 1 hour/month | measurement | 1 |

## 10. Tooling (evaluated 2026-10-01)

Two candidate tools were reviewed. They sit at different layers and are not
substitutes for each other.

| | [claude-seo](https://github.com/AgriciDaniel/claude-seo) | [open-seo](https://github.com/every-app/open-seo) |
|---|---|---|
| What it is | SEO audit + generation **workflow** for Claude Code: 26 sub-skills, 19 agents, `/seo audit`, schema, sitemap, content briefs, GEO scoring, drift | SEO **data platform** (Semrush/Ahrefs alternative): keyword research, rank tracking, backlinks, competitor insights, site audits, AI visibility |
| Agent compatibility | Claude Code plugin (also a Codex port). Not an OpenCode plugin | Exposes an MCP server — works with any MCP-capable agent, including OpenCode |
| Data | Local by default (fetches the sites you point it at); richer via MCP extensions (Ahrefs, SE Ranking, Profound, DataForSEO) | DataForSEO API behind everything |
| Cost | Free (MIT) + your Claude Code subscription | MIT, self-host free (Docker/Cloudflare); DataForSEO is pay-per-request; hosted $10/mo |
| Best for | One-off deep audits, schema/content generation, AI-search scoring | Ongoing keyword/SERP/backlink data, rank + AI-visibility tracking |

**Decision: adopt open-seo for data; borrow claude-seo as reference material;
install neither as a hard dependency right now.**

- The bottleneck for VeriScript is execution of Phases 0–1 (surface, pages,
  distribution), which neither tool does for us. We can implement those
  directly in this repo.
- The one thing we cannot self-generate is **data**: what the queries
  actually are, who ranks, what we rank for, whether AI engines cite us.
  open-seo (via its MCP server) plugs into this agent harness and covers
  Phase 7's measurement loop — including AI-visibility tracking — without a
  subscription. Self-host free or use the hosted tier; the data layer under
  it is DataForSEO.
- **DataForSEO cost reality (verified 2026-10-01):** the $1 trial credit
  does not unlock data endpoints — the account must complete verification
  first, and where phone verification is unavailable the unlock is the
  **$50 one-time minimum deposit** (credits never expire). Treat $50 as the
  entry cost, not the $1 trial. Deferred for now: the site has no indexable
  pages yet, GSC will give first-party query data once pages ship, and the
  AI-citation baseline can run manually for $0. Revisit when rank tracking
  and mention monitoring have something to measure.
- claude-seo's audit workflow largely duplicates what we can do directly
  (and with GSC/Bing/Lighthouse, all free), and requires a Claude Code
  toolchain we don't otherwise need. Its reference material is excellent —
  particularly the Google-currency ledger and the llms.txt evidence review —
  and can be read without installing anything. Worth revisiting if Claude
  Code becomes part of the workflow.
- Neither tool replaces the off-site half of the plan (§2.3, §5): mentions,
  benchmark citations, launches, directories. Tools point; distribution
  moves.

## Sources

- [1] DevToolLab, *Generative Engine Optimization for Developers* (2026-06-10) — AI-query share, JS rendering, freshness
- [2] Noble, *Is AEO Worth Investing In?* (2026-04-13) — Ahrefs 75k-brand mention study, 85% third-party citations, 6.5×, ChatGPT 900M users
- [3] Seer Interactive (via Naturaily, *AEO: How to Get Cited*, 2026-08-03) — comparison 95% / question 86% AI Overview trigger rates
- [4] AirOps, *AEO: Complete Guide 2026* — citation vs mention, off-site weighting
- [5] DevToolLab + Google Search Central (via Search Engine Roundtable, 2026-06-16) — crawler allowlists, FAQ rich results removed, schema for AI extraction
- [6] LLMS.txt evidence round-up: Ahrefs (97% zero requests, May 2026), SE Ranking (~10% adoption, no correlation), Google Search Central (ignored by Google), The AC Group (408/515M AI-bot visits) — 2026
- arXiv:2606.29213, *Can OCR-VLMs Read Devanagari?* (2026-06-28) — the research-side gap this project can fill
- Competitor pages audited 2026-10-01: fastocr.org/nepali-ocr, olocr.com/nepali-ocr, toolghar.com/nepali-handwriting-to-text, puspachaulagai.com.np/tools/ocr-reader
- [7] AgriciDaniel/claude-seo — Google-currency reference: FAQ rich results retired 2026-05-07, HowTo retired 2023, llms.txt evidence review (README + `skills/seo-schema/references/deprecated-types-2024-2026.md`), 2026-09
- Tooling reviewed 2026-10-01: github.com/AgriciDaniel/claude-seo (18.1k★, MIT), github.com/every-app/open-seo (22k★, MIT)
