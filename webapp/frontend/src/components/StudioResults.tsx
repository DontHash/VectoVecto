import { createMemo, For, Show, createSignal } from "solid-js";
import { CodeBlock } from "./CodeBlock";
import { CompareSlider } from "./CompareSlider";
import { CorrectionReview } from "./CorrectionReview";
import { type StudioState } from "../lib/studio-state";

type Tab = "compare" | "queue" | "transcript" | "files";

/** Everything a finished run produces, organised in tabs. */
export function StudioResults(props: { state: StudioState }) {
  const s = props.state;
  const [tab, setTab] = createSignal<Tab>("compare");

  const afterUrl = createMemo(() => {
    const p = s.result();
    if (!p) return null;
    return s.showBoxes() && p.files.overlay ? p.files.overlay : p.files.restored;
  });

  const tabs = createMemo<[Tab, string][]>(() => {
    const n = s.result()?.review.length ?? 0;
    return [
      ["compare", "compare"],
      ["queue", `review queue (${n})`],
      ["transcript", "transcript"],
      ["files", "files"],
    ];
  });

  return (
    <Show when={s.result()}>
      {(result) => (
        <div class="results__panel">
          <div class="stats">
            <div>
              <div class="stat__label">time</div>
              <div class="stat__value">{result().meta.seconds.toFixed(1)}s</div>
            </div>
            <div>
              <div class="stat__label">engine</div>
              <div class="stat__value">{result().meta.backend}</div>
            </div>
            <div>
              <div class="stat__label">tokens</div>
              <div class="stat__value">{result().meta.n_tokens}</div>
            </div>
            <div>
              <div class="stat__label">flagged</div>
              <div class="stat__value stat__value--accent">{result().meta.flagged}</div>
            </div>
            <div>
              <div class="stat__label">digit conflicts</div>
              <div class="stat__value">{result().meta.digit_conflicts}</div>
            </div>
            <Show when={result().meta.pages_processed > 1}>
              <div>
                <div class="stat__label">pages</div>
                <div class="stat__value">{result().meta.pages_processed}</div>
              </div>
            </Show>
          </div>

          <Show when={result().meta.resized}>
            <p class="field__hint">
              Large page: downscaled to 2500 px before processing; the review boxes
              match the processed size.
            </p>
          </Show>
          <Show when={result().meta.pages_capped}>
            <p class="field__hint">
              Only the first {result().meta.page_cap} pages were processed — split
              larger PDFs into parts.
            </p>
          </Show>

          <div class="tabs" role="tablist">
            <For each={tabs()}>
              {([id, label]) => (
                <button
                  class="tab"
                  role="tab"
                  aria-selected={tab() === id}
                  onClick={() => setTab(id)}
                >
                  {label}
                </button>
              )}
            </For>
          </div>

          <Show when={tab() === "compare"}>
            <div class="stack-16">
              <label class="checkbox">
                <input
                  type="checkbox"
                  checked={s.showBoxes()}
                  onChange={(e) => s.setShowBoxes(e.currentTarget.checked)}
                />
                <span>show the review boxes</span>
              </label>
              <Show
                when={s.inputUrl()}
                fallback={
                  <div class="card card--flush">
                    <img
                      class="compare__img"
                      src={result().files.overlay ?? result().files.restored}
                      alt="Restored page with review overlay"
                    />
                  </div>
                }
              >
                <CompareSlider
                  before={s.inputUrl()!}
                  after={afterUrl()!}
                  alt="restored page"
                  beforeLabel="input"
                  afterLabel={s.showBoxes() ? "review overlay" : "restored"}
                />
              </Show>
              <p class="field__hint">
                The overlay is the pipeline's own output: a box per token, coloured by the flag.
                Alternative readings sit above conflicting digits.
              </p>
            </div>
          </Show>

          <Show when={tab() === "queue"}>
            <div class="stack-16">
              <CorrectionReview state={s} />
              <p class="field__hint">
                Confirm or fix each queued token; the corrected artifacts and
                the optional training export appear here. Corrections stay in
                this run until you download them.
              </p>
            </div>
          </Show>

          <Show when={tab() === "transcript"}>
            <div class="stack-16">
              <pre class="transcript">{result().transcript || "— no text —"}</pre>
              <CodeBlock label="copy transcript" code={result().transcript} />
            </div>
          </Show>

          <Show when={tab() === "files"}>
            <div class="stack-16">
              <div class="results__actions">
                <a class="download" href={`${result().files.restored}?download=1`}>
                  ↓ restored.png
                </a>
                <Show when={result().files.overlay}>
                  <a class="download" href={`${result().files.overlay}?download=1`}>
                    ↓ overlay.png
                  </a>
                </Show>
                <Show when={result().files.pdf}>
                  <a class="download" href={`${result().files.pdf}?download=1`}>
                    ↓ searchable.pdf
                  </a>
                </Show>
                <Show when={result().files.combined_pdf}>
                  <a class="download" href={`${result().files.combined_pdf}?download=1`}>
                    ↓ combined.pdf
                  </a>
                </Show>
                <Show when={result().files.txt}>
                  <a class="download" href={`${result().files.txt}?download=1`}>
                    ↓ transcript.txt
                  </a>
                </Show>
                <Show when={result().files.md}>
                  <a class="download" href={`${result().files.md}?download=1`}>
                    ↓ transcript.md
                  </a>
                </Show>
                <Show when={result().files.combined_txt}>
                  <a class="download" href={`${result().files.combined_txt}?download=1`}>
                    ↓ combined.txt
                  </a>
                </Show>
                <Show when={result().files.combined_md}>
                  <a class="download" href={`${result().files.combined_md}?download=1`}>
                    ↓ combined.md
                  </a>
                </Show>
                <Show when={result().files.json}>
                  <a class="download" href={`${result().files.json}?download=1`}>
                    ↓ ocr.json
                  </a>
                </Show>
              </div>
              <p class="field__hint">
                run <span class="mono">{result().run_id}</span> · {result().status_line} · files
                are deleted after the retention window.
              </p>
            </div>
          </Show>
        </div>
      )}
    </Show>
  );
}
