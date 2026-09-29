import { createSignal, onCleanup, onMount, Show } from "solid-js";
import { formatBytes } from "../lib/api";

const ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf";

/** Drag & drop, click, or paste one page. */
export function DropZone(props: {
  file: File | null;
  onFile: (file: File) => void;
  onClear: () => void;
  disabled?: boolean;
}) {
  const [dragging, setDragging] = createSignal(false);
  let input: HTMLInputElement | undefined;
  let thumbUrl: string | undefined;

  const accept = (file: File | undefined | null) => {
    if (!file || props.disabled) return;
    props.onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer?.files?.[0]);
  };

  onMount(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/"),
      );
      const file = item?.getAsFile();
      if (file) accept(file);
    };
    document.addEventListener("paste", onPaste);
    onCleanup(() => {
      document.removeEventListener("paste", onPaste);
      if (thumbUrl) URL.revokeObjectURL(thumbUrl);
    });
  });

  return (
    <div>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        class="visually-hidden"
        onChange={(e) => accept(e.currentTarget.files?.[0])}
      />

      <Show
        when={props.file}
        fallback={
          <div
            class="dropzone"
            data-drag={dragging()}
            role="button"
            tabindex={0}
            aria-label="Upload a page: click, drop a file, or paste an image"
            onClick={() => input?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                input?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <span class="dropzone__icon" aria-hidden="true">
              ¶
            </span>
            <span class="dropzone__title">Drop a page here, click to browse, or paste</span>
            <span class="dropzone__hint">png · jpg · webp · tiff · pdf</span>
          </div>
        }
      >
        {(file) => {
          if (thumbUrl) URL.revokeObjectURL(thumbUrl);
          thumbUrl = file().type.startsWith("image/") ? URL.createObjectURL(file()) : undefined;
          return (
            <div class="fileticket">
              <Show when={thumbUrl}>
                <img class="fileticket__thumb" src={thumbUrl} alt="" />
              </Show>
              <div class="fileticket__meta">
                <div class="fileticket__name">{file().name}</div>
                <div class="fileticket__size">{formatBytes(file().size)}</div>
              </div>
              <button
                class="fileticket__clear"
                type="button"
                aria-label="Remove file"
                onClick={props.onClear}
              >
                ✕
              </button>
            </div>
          );
        }}
      </Show>
    </div>
  );
}
