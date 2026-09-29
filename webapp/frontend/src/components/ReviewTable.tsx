import { For, Show } from "solid-js";
import { flagMeta, riskTone, rowTone } from "../lib/flags";
import type { QueueItem } from "../lib/types";

/** The review queue, rendered with the product's own flag semantics. */
export function ReviewTable(props: { rows: QueueItem[]; caption?: string }) {
  return (
    <div>
      <table class="table">
        <thead>
          <tr>
            <th>token</th>
            <th>alternative kept</th>
            <th>why it is queued</th>
            <th>conf</th>
            <th>risk</th>
          </tr>
        </thead>
        <tbody>
          <For each={props.rows}>
            {(row) => {
              const tone = rowTone(row.flags);
              const altClass =
                tone === "red" ? "alt alt--red" : tone === "amber" ? "alt alt--amber" : "alt";
              return (
                <tr>
                  <td>
                    <span class="tok">{row.text}</span>
                  </td>
                  <td>
                    <Show when={row.alt_text} fallback={<span class="muted">—</span>}>
                      <span class={altClass}>{row.alt_text}</span>
                    </Show>
                  </td>
                  <td>
                    <span style={{ display: "inline-flex", "flex-wrap": "wrap", gap: "8px" }}>
                      <For each={row.flags}>
                        {(flag) => {
                          const meta = flagMeta(flag);
                          return (
                            <span class={`chip chip--${meta.tone}`}>
                              <span class="chip__dot" />
                              {meta.label}
                            </span>
                          );
                        }}
                      </For>
                    </span>
                  </td>
                  <td>
                    <span class="num">{row.conf === null ? "—" : row.conf.toFixed(1)}</span>
                  </td>
                  <td>
                    <span class="risk">
                      <span class="risk__bar" aria-hidden="true">
                        <span
                          class={
                            "risk__fill" +
                            (riskTone(row.risk) === "high"
                              ? " risk__fill--high"
                              : riskTone(row.risk) === "mid"
                                ? " risk__fill--mid"
                                : "")
                          }
                          style={{ width: `${Math.min(100, ((row.risk ?? 0) / 7) * 100)}%` }}
                        />
                      </span>
                      <span class="num">{row.risk === null ? "—" : row.risk.toFixed(1)}</span>
                    </span>
                  </td>
                </tr>
              );
            }}
          </For>
        </tbody>
      </table>
      <Show when={props.caption}>
        <p class="eyebrow" style={{ "margin-top": "16px" }}>
          {props.caption}
        </p>
      </Show>
    </div>
  );
}
