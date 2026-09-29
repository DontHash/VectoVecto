import { createSignal, onCleanup, onMount } from "solid-js";

/**
 * Before/after compare with a drag handle. Pointer-drag anywhere on the
 * surface, arrow keys on the handle. Mirrors the restore wipe from the film.
 */
export function CompareSlider(props: {
  before: string;
  after: string;
  alt: string;
  beforeLabel?: string;
  afterLabel?: string;
  /** initial handle position, 0–100 */
  initial?: number;
  onPositionChange?: (p: number) => void;
}) {
  const [pos, setPos] = createSignal(props.initial ?? 50);
  const [dragging, setDragging] = createSignal(false);
  let surface: HTMLDivElement | undefined;

  const clamp = (v: number) => Math.min(100, Math.max(0, v));

  const setFromClientX = (clientX: number) => {
    if (!surface) return;
    const rect = surface.getBoundingClientRect();
    const next = clamp(((clientX - rect.left) / rect.width) * 100);
    setPos(next);
    props.onPositionChange?.(next);
  };

  const onPointerDown = (e: PointerEvent) => {
    setDragging(true);
    surface?.setPointerCapture(e.pointerId);
    setFromClientX(e.clientX);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (dragging()) setFromClientX(e.clientX);
  };
  const onPointerUp = (e: PointerEvent) => {
    setDragging(false);
    surface?.releasePointerCapture(e.pointerId);
  };

  const onKeyDown = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === "ArrowLeft") {
      setPos((p) => clamp(p - step));
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      setPos((p) => clamp(p + step));
      e.preventDefault();
    } else if (e.key === "Home") {
      setPos(0);
      e.preventDefault();
    } else if (e.key === "End") {
      setPos(100);
      e.preventDefault();
    }
  };

  onMount(() => {
    const stop = () => setDragging(false);
    window.addEventListener("pointerup", stop);
    onCleanup(() => window.removeEventListener("pointerup", stop));
  });

  return (
    <div
      ref={surface}
      class="compare"
      data-dragging={dragging()}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <img class="compare__img" src={props.after} alt={`${props.alt} — restored`} draggable={false} />
      <div
        class="compare__before"
        style={{ "clip-path": `inset(0 ${100 - pos()}% 0 0)` }}
        aria-hidden="true"
      >
        <img class="compare__img" src={props.before} alt="" draggable={false} />
      </div>

      <span class="compare__tag compare__tag--left">{props.beforeLabel ?? "input"}</span>
      <span class="compare__tag compare__tag--right">{props.afterLabel ?? "overlay"}</span>

      <div
        class="compare__handle"
        style={{ left: `${pos()}%` }}
        role="slider"
        tabindex={0}
        aria-label="Compare input and output"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos())}
        onKeyDown={onKeyDown}
      >
        <span class="compare__grip" aria-hidden="true">
          ↔↕
        </span>
      </div>
    </div>
  );
}
