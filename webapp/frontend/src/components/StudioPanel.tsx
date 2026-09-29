import { For, Show } from "solid-js";
import { DropZone } from "./DropZone";
import { EXAMPLE_FILES, limitsLine, type StudioState } from "../lib/studio-state";
import type { Lang, OcrEngine } from "../lib/types";

/** The studio controls: drop a page, pick an example, set options, run.
 *  `compact` is the homepage hero treatment. */
export function StudioPanel(props: { state: StudioState; compact?: boolean }) {
  const s = props.state;
  const running = () => s.phase() === "running";

  return (
    <div class="panel">
      <DropZone
        file={s.file()}
        onFile={s.chooseFile}
        onClear={s.clearFile}
        disabled={running()}
      />

      <div class="panel__examples">
        <span class="eyebrow">or try a real example</span>
        <div class="example-chips">
          <For each={EXAMPLE_FILES}>
            {(ex) => (
              <button
                class="btn btn--ghost btn--small"
                type="button"
                disabled={running()}
                onClick={() => void s.pickExample(ex.path, ex.name)}
              >
                {ex.label}
              </button>
            )}
          </For>
        </div>
      </div>

      <div class="panel__options">
        <div class="field">
          <label class="field__label" for={`lang-${props.compact ? "hero" : "page"}`}>
            Language
          </label>
          <select
            id={`lang-${props.compact ? "hero" : "page"}`}
            class="select"
            value={s.lang()}
            disabled={running()}
            onChange={(e) => s.setLang(e.currentTarget.value as Lang)}
          >
            <option value="ne">Nepali (Devanagari)</option>
            <option value="hi">Hindi (Devanagari)</option>
            <option value="en">English / Latin</option>
          </select>
        </div>

        <label class="checkbox">
          <input
            type="checkbox"
            checked={s.deskew()}
            disabled={running()}
            onChange={(e) => s.setDeskew(e.currentTarget.checked)}
          />
          <span>Deskew small rotations</span>
        </label>

        <button
          class="btn btn--ghost btn--small"
          type="button"
          disabled={running()}
          onClick={() => s.setAdvancedOpen(!s.advancedOpen())}
        >
          {s.advancedOpen() ? "hide options" : "more options"}
        </button>
      </div>

      <Show when={s.advancedOpen()}>
        <div class="stack-16">
          <div class="field">
            <label
              class="field__label"
              for={`ocr-${props.compact ? "hero" : "page"}`}
            >
              OCR engine
            </label>
            <select
              id={`ocr-${props.compact ? "hero" : "page"}`}
              class="select"
              value={s.ocr()}
              disabled={running()}
              onChange={(e) => s.setOcr(e.currentTarget.value as OcrEngine)}
            >
              <option value="auto">Auto (best available)</option>
              <option value="rapidocr">RapidOCR (recommended)</option>
              <option value="tesseract">Tesseract (optional install)</option>
            </select>
          </div>

          <label class="checkbox">
            <input
              type="checkbox"
              checked={s.autoRotate()}
              disabled={running()}
              onChange={(e) => s.setAutoRotate(e.currentTarget.checked)}
            />
            <span>Auto-rotate sideways / upside-down pages</span>
          </label>

          <label class="checkbox">
            <input
              type="checkbox"
              checked={s.allPages()}
              disabled={running()}
              onChange={(e) => s.setAllPages(e.currentTarget.checked)}
            />
            <span>All pages of a PDF (first page by default)</span>
          </label>

          <div class="field">
            <span class="field__label">Files to write</span>
            <div class="example-chips">
              <label class="checkbox">
                <input
                  type="checkbox"
                  checked={s.wantOverlay()}
                  disabled={running()}
                  onChange={(e) => s.setWantOverlay(e.currentTarget.checked)}
                />
                <span>overlay</span>
              </label>
              <label class="checkbox">
                <input
                  type="checkbox"
                  checked={s.wantPdf()}
                  disabled={running()}
                  onChange={(e) => s.setWantPdf(e.currentTarget.checked)}
                />
                <span>searchable PDF</span>
              </label>
              <label class="checkbox">
                <input
                  type="checkbox"
                  checked={s.wantTxt()}
                  disabled={running()}
                  onChange={(e) => s.setWantTxt(e.currentTarget.checked)}
                />
                <span>transcript</span>
              </label>
              <label class="checkbox">
                <input
                  type="checkbox"
                  checked={s.wantMd()}
                  disabled={running()}
                  onChange={(e) => s.setWantMd(e.currentTarget.checked)}
                />
                <span>markdown</span>
              </label>
            </div>
          </div>
        </div>
      </Show>

      <Show when={!props.compact}>
        <p class="field__hint">
          Picking the wrong script lowers accuracy — it is not auto-detected per token.
        </p>
      </Show>

      <div class="runbar">
        <button
          class="btn btn--primary btn--block"
          type="button"
          disabled={!s.file() || running()}
          onClick={() => void s.start()}
        >
          <Show when={!running()} fallback={<>Reading… {s.elapsed().toFixed(1)}s</>}>
            Restore &amp; read <span class="arrow">→</span>
          </Show>
        </button>
        <Show when={running()}>
          <div class="progress" aria-hidden="true">
            <div class="progress__fill" />
          </div>
        </Show>
        <div class="runbar__status">
          <span>{limitsLine(s.health())}</span>
          <Show when={s.health()?.busy}>
            <span>worker busy — you may queue</span>
          </Show>
        </div>
      </div>

      <Show when={s.phase() === "error" && s.error()}>
        <div class="alert" role="alert">
          {s.error()}
        </div>
      </Show>
    </div>
  );
}
