import { For } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { CompareSlider } from "../components/CompareSlider";

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
    note: "A clean, machine-printed page — every token accepted. This is the easy case; the hard ones are what the review queue is for.",
  },
];

export function Landing() {
  return (
    <>
      {/* ------------------------------------------------ hero */}
      <section class="hero" id="studio">
        <div class="container hero__grid">
          <div class="hero__copy">
            <span class="eyebrow">Offline document restoration — Devanagari first</span>
            <h1 class="display hero__title">It will not invent the numbers on your bill.</h1>
            <p class="lede">
              VeriScript turns photos, scans and PDFs into cleaned pages, searchable PDFs and
              Markdown — and puts every uncertain number in a review queue instead of guessing.
            </p>
            <div class="hero__cta">
              <a class="btn btn--primary" href="/studio">Try it</a>
              <a class="textlink" href="#examples">See real examples ↓</a>
            </div>
            <div class="hero__meta">
              <span>no signup</span>
              <span>no account</span>
              <span>files deleted after the demo</span>
              <span>mit licensed</span>
            </div>
          </div>

          <div class="panel-card">
            <div class="panel-card__head">
              <span class="eyebrow">Try it — the studio</span>
              <span class="eyebrow">one page · seconds</span>
            </div>
            <div class="panel-card__body">
              <p class="lede">
                Drop a page or PDF and see the full product run: the cleaned page, the numbered
                review overlay, the review queue with alternative readings kept, and the
                downloads — searchable PDF, transcript, Markdown, OCR JSON.
              </p>
              <a class="btn btn--primary btn--block" href="/studio">Try it — open the Studio</a>
              <p class="eyebrow">No signup. No account. Nothing to install for the demo.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------ examples */}
      <section class="section" id="examples">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">Examples</span>
            <h2 class="h2">Real pages. Real output. Drag the handle.</h2>
            <p class="lede">
              These images are not mockups — they are the pipeline's own output, including the
              cases it flags.
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

      {/* ------------------------------------------------ offline */}
      <section class="section slate-section" id="offline">
        <div class="container">
          <div class="section__head">
            <span class="eyebrow">Offline by design</span>
            <h2 class="h2">This page is a demo. The product runs on your machine.</h2>
            <p class="lede">
              The demo processes your page on this server and deletes it after the retention
              window. The product itself has no cloud calls, no telemetry and no account —
              install once and it stays yours.
            </p>
          </div>

          <CodeBlock
            label="run it locally"
            code={`pip install -r requirements.txt

# one page in — cleaned page, searchable PDF, transcript, Markdown, OCR JSON out
python -m veriscript --mode document --input page.jpg --output out/run1

# the same studio, locally
python webapp/server.py   # → http://127.0.0.1:8000`}
          />

          <p class="eyebrow" style={{ "margin-top": "24px" }}>
            the pipeline runs with networking disabled — nothing reaches out
          </p>
        </div>
      </section>
    </>
  );
}
