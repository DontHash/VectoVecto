import { Show } from "solid-js";
import { SheetStage } from "../components/SheetStage";
import { StudioPanel } from "../components/StudioPanel";
import { StudioResults } from "../components/StudioResults";
import { createStudioState } from "../lib/studio-state";

export function Studio() {
  const studio = createStudioState();

  return (
    <div class="studio">
      <div class="container">
        <div class="studio__head">
          <span class="eyebrow">Studio</span>
          <h1 class="h2">Restore a page — and watch it show its work.</h1>
          <p class="lede">
            One page at a time. Processing happens on this server for the demo; the app itself is
            built to run offline, on your machine.
          </p>
        </div>

        <div class="studio__grid">
          <div class="studio__panel">
            <StudioPanel state={studio} />
          </div>

          <div class="results">
            <Show
              when={studio.phase() === "done" && studio.result()}
              fallback={<IdleStage />}
            >
              <StudioResults state={studio} />
            </Show>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Empty state: the desk plus what a run will produce. */
function IdleStage() {
  return (
    <div class="stack-16">
      <div class="stage-card">
        <div class="stage-card__canvas">
          <SheetStage
            pages={["/examples/letterpress-hero.jpg", "/examples/invoice-hero.jpg"]}
            alts={[
              "1955 letterpress page, real scan",
              "Invoice photo under a simulated phone shadow",
            ]}
          />
        </div>
        <div class="stage-card__foot">real pipeline input — nothing staged</div>
      </div>

      <div class="expect">
        <span class="eyebrow">What you'll get</span>
        <ol>
          <li>the cleaned page, side by side with your input</li>
          <li>the numbered review overlay — every uncertain token boxed</li>
          <li>the review queue with alternative readings kept</li>
          <li>searchable PDF · transcript · OCR JSON — all downloadable</li>
        </ol>
        <p class="field__hint">
          The first run on a fresh server also warms the OCR engine, so it can take a little
          longer.
        </p>
      </div>
    </div>
  );
}
