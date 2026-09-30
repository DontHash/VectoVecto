import { createEffect, For, Show } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { CompareSlider } from "../components/CompareSlider";
import { ReviewTable } from "../components/ReviewTable";
import { StudioPanel } from "../components/StudioPanel";
import { StudioResults } from "../components/StudioResults";
import { createStudioState } from "../lib/studio-state";
import { LETTERPRESS_META, LETTERPRESS_QUEUE } from "../lib/queue-sample";

const EXAMPLES = [
  {
    id: "letterpress",
    title: "1955 letterpress page",
    before: "/examples/letterpress-input.jpg",
    after: "/examples/letterpress-overlay.jpg",
    meta: ["heidata · cc by 4.0", "30 tokens · 17 flagged"],
    note: "Red boxes are digit conflicts; the alternative reading is kept beside the box. Green means accepted, amber means uncertain.",
  },
  {
    id: "invoice",
    title: "Invoice photo",
    before: "/examples/invoice-input.jpg",
    after: "/examples/invoice-overlay.jpg",
    meta: ["synthetic fixture", "27 tokens · 8 flagged · 4 conflicts"],
    note: "A simulated phone shadow over the repo's own invoice. The restore stream removes the shadow; the flags stay exactly where the reading was unsure.",
  },
  {
    id: "clean",
    title: "Clean invoice",
    before: "/examples/synthetic-invoice.jpg",
    after: "/examples/synthetic-invoice-overlay.jpg",
    meta: ["clean render", "27 tokens · 0 flagged"],
    note: "A clean, machine-printed page — every token accepted. This is the easy case; the film is about the hard ones.",
  },
];

const STEPS = [
  {
    title: "Orientation",
    body: "EXIF is honoured; 0/90/180/270 is decided from OCR evidence on the upright candidates, not by guessing. 90/90 rotated pages in the frozen set were decided; 0 upright pages were falsely rotated.",
  },
  {
    title: "Layout & reading order",
    body: "Line boxes, columns and tables are found before transcription. Two-column pages come out in the right order — WER 0.870 → 0.268 on the arXiv split-page set.",
  },
  {
    title: "Recognition",
    body: "RapidOCR (PP-OCRv6) reads Devanagari, Hindi and English by default; Tesseract 5 is an optional second engine. The Latin model ships with the app; the Devanagari model downloads once on first use, then everything runs offline.",
  },
  {
    title: "Risk ranking",
    body: "Digits are the highest-stakes tokens, so they are re-read and risk-scored, and confidence is calibrated (isotonic) instead of taken at face value.",
  },
  {
    title: "Review queue & exports",
    body: "Uncertain tokens ship as a review queue with their alternative readings kept — next to a searchable PDF, the numbered overlay, the transcript and the JSON.",
  },
];

const LIMITS = [
  "Letterpress Devanagari is the honest bar: raw page CER is 0.434, and the opt-in line reader brings it to 0.253 — still not clean.",
  "Phone photos are measured on a synthetic proxy (simulated shadow, blur, noise), not on a real field set.",
  "A vision-language reader was evaluated and does not ship: 152 s/page and 224 invented tokens failed its cost gate.",
  "The queue — not the colors — is the honest signal: token flags carry little information on real Devanagari (ECE 0.82).",
];

const EVIDENCE = [
  {
    metric: "CER 0.135 [0.097–0.186]",
    label: "Modern government PDFs · exact-token recall 0.877",
    note: "41 born-digital pages, Gemini-2.5-Pro anchor, engine-corroborated",
  },
  {
    metric: "digit recall@10 0.88",
    label: "Letterpress review queue · n=69 frozen pages",
    note: "the share of true digit errors that reach the queue",
  },
  {
    metric: "16/16 · 90/90",
    label: "Orientation and rotation decisions",
    note: "synthetic set upright; rotated pages decided; 0 false rotations on upright scans",
  },
  {
    metric: "~4 s/page",
    label: "One laptop CPU, offline",
    note: "22 s for the first page including model warm-up; every number reproducible from the repo",
  },
];

export function Landing() {
  const studio = createStudioState();
  let resultRef: HTMLDivElement | undefined;

  createEffect(() => {
    if (studio.phase() !== "done") return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    requestAnimationFrame(() =>
      resultRef?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" }),
    );
  });

  return (
    <>
      {/* ------------------------------------------------ hero = the product */}
      <section class="hero" id="studio">
        <div class="container hero__grid">
          <div class="hero__copy">
            <span class="eyebrow">Offline document restoration — Devanagari first</span>
            <h1 class="display hero__title">It will not invent the numbers on your bill.</h1>
            <p class="lede">
              VeriScript restores photos and scans into cleaned pages and searchable PDFs — and
              puts every uncertain digit in a measured review queue instead of guessing.
            </p>
            <a class="textlink" href="#examples">
              See real examples ↓
            </a>
            <div class="hero__meta">
              <span>nepali · hindi · english</span>
              <span>measured on frozen sets</span>
              <span>mit licensed</span>
            </div>
          </div>

          <div class="panel-card">
            <div class="panel-card__head">
              <span class="eyebrow">Studio — try it now</span>
              <span class="eyebrow">one page · seconds</span>
            </div>
            <StudioPanel state={studio} compact />
          </div>
        </div>

        <Show when={studio.phase() === "done" && studio.result()}>
          <div class="container">
            <div class="results-section" ref={resultRef}>
              <div class="results-section__head">
                <span class="eyebrow">Your page — restored, read, risk-ranked</span>
              </div>
              <StudioResults state={studio} />
            </div>
          </div>
        </Show>
      </section>

      {/* ------------------------------------------------ examples */}
      <section class="section" id="examples">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">Examples</span>
            <h2 class="h2">Real pages. Real output. Drag the handle.</h2>
            <p class="lede">
              These images are not mockups — they are the pipeline's own output, including the
              cases it flags. Every queue on this page is read from a run's{" "}
              <span class="mono">ocr.json</span>.
            </p>
          </div>

          <div class="examples">
            <For each={EXAMPLES}>
              {(ex) => (
                <article class="example">
                  <div class="example__head">
                    <h3 class="h3">{ex.title}</h3>
                    <div class="example__meta">
                      <For each={ex.meta}>{(m) => <span>{m}</span>}</For>
                    </div>
                  </div>
                  <CompareSlider before={ex.before} after={ex.after} alt={ex.title} />
                  <p class="example__note">{ex.note}</p>
                </article>
              )}
            </For>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------ pipeline */}
      <section class="section" id="pipeline">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">How it reads</span>
            <h2 class="h2">Five steps, and the fifth one is the point.</h2>
          </div>

          <div class="pipeline">
            <ol class="steps">
              <For each={STEPS}>
                {(step) => (
                  <li>
                    <div>
                      <h3>{step.title}</h3>
                      <p>{step.body}</p>
                    </div>
                  </li>
                )}
              </For>
            </ol>

            <aside class="honest">
              <h3>What it won't do — on purpose</h3>
              <ul>
                <For each={LIMITS}>{(limit) => <li>{limit}</li>}</For>
              </ul>
            </aside>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------ queue */}
      <section class="section" id="queue">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">The review queue</span>
            <h2 class="h2">The queue is the product.</h2>
            <p class="lede">
              A real queue from the letterpress run: the nine highest-risk rows of seventeen. The
              rest ship in every run's <span class="mono">ocr.json</span>.
            </p>
          </div>

          <div class="queue__grid">
            <div class="legend">
              <div class="legend__item">
                <span class="legend__swatch legend__swatch--green" />
                <div>
                  <strong>Accepted</strong>
                  <p>The pipeline stands behind the reading.</p>
                </div>
              </div>
              <div class="legend__item">
                <span class="legend__swatch legend__swatch--amber" />
                <div>
                  <strong>Uncertain</strong>
                  <p>Low confidence or an uncertain digit — look at it.</p>
                </div>
              </div>
              <div class="legend__item">
                <span class="legend__swatch legend__swatch--red" />
                <div>
                  <strong>Digit conflict</strong>
                  <p>
                    Two readings disagree; the alternative is kept in the queue — never silently
                    replaced.
                  </p>
                </div>
              </div>
              <p class="eyebrow" style={{ "margin-top": "16px" }}>
                the queue, not the colors, is the honest signal
              </p>
            </div>

            <div>
              <ReviewTable
                rows={LETTERPRESS_QUEUE.slice(0, 9)}
                caption={`${LETTERPRESS_META.page} — ${LETTERPRESS_META.tokens} tokens · ${LETTERPRESS_META.flagged} flagged · ${LETTERPRESS_META.source}`}
              />
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------ evidence */}
      <section class="section" id="evidence">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">Evidence</span>
            <h2 class="h2">Measured, not asserted.</h2>
          </div>

          <table class="evidence-table">
            <tbody>
              <For each={EVIDENCE}>
                {(row) => (
                  <tr>
                    <th scope="row">
                      <span class="metric">{row.metric}</span>
                      {row.label}
                    </th>
                    <td>{row.note}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>

          <p class="eyebrow" style={{ "margin-top": "24px" }}>
            frozen content-hash sets · bootstrap 95% CIs · reproduction commands and known
            limitations in docs/EVALUATION.md
          </p>
        </div>
      </section>

      {/* ------------------------------------------------ offline */}
      <section class="section slate-section" id="offline">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">Offline by design</span>
            <h2 class="h2">This page is a demo. The product runs on your machine.</h2>
            <p class="lede">
              The studio processes your page on this server and deletes it after the retention
              window. The real guarantee — and the reason VeriScript exists — is that the pipeline
              runs offline: no cloud calls, no telemetry, no account. Install it and it stays yours.
            </p>
          </div>

          <CodeBlock
            label="copy commands"
            code={`pip install -r requirements.txt

# one page in — searchable PDF, overlay, transcript, JSON out
python cli.py --mode document --input page.jpg --output out/run1

# a local studio, same pipeline
python app.py        # http://127.0.0.1:7860`}
          />

          <p class="eyebrow" style={{ "margin-top": "24px" }}>
            the pipeline runs with networking disabled — nothing reaches out
          </p>
        </div>
      </section>
    </>
  );
}
