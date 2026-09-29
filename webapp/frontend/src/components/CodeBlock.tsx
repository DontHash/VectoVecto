import { createSignal } from "solid-js";

/** Mono block with a copy-to-clipboard affordance. */
export function CodeBlock(props: { code: string; label?: string }) {
  const [copied, setCopied] = createSignal(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(props.code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard may be unavailable; the block stays selectable */
    }
  };

  return (
    <div class="codeblock">
      <button class="copy" type="button" onClick={copy} aria-label="Copy commands">
        {copied() ? "copied" : (props.label ?? "copy")}
      </button>
      <pre>
        <code>{props.code}</code>
      </pre>
    </div>
  );
}
