/** Studio state — one factory, used by the homepage hero and the full studio
 *  page, so the product behaves identically wherever it is embedded. */
import { createSignal, onCleanup } from "solid-js";
import { fetchHealth, runRestore } from "./api";
import type { Health, Lang, OcrEngine, RunPayload } from "./types";

export type StudioPhase = "idle" | "running" | "done" | "error";

export function createStudioState() {
  const [file, setFile] = createSignal<File | null>(null);
  const [lang, setLang] = createSignal<Lang>("ne");
  const [deskew, setDeskew] = createSignal(false);
  const [ocr, setOcr] = createSignal<OcrEngine>("auto");
  const [allPages, setAllPages] = createSignal(false);
  const [autoRotate, setAutoRotate] = createSignal(true);
  const [wantOverlay, setWantOverlay] = createSignal(true);
  const [wantPdf, setWantPdf] = createSignal(true);
  const [wantTxt, setWantTxt] = createSignal(true);
  const [wantMd, setWantMd] = createSignal(true);
  const [advancedOpen, setAdvancedOpen] = createSignal(true);
  const [phase, setPhase] = createSignal<StudioPhase>("idle");
  const [result, setResult] = createSignal<RunPayload | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [elapsed, setElapsed] = createSignal(0);
  const [health, setHealth] = createSignal<Health | null>(null);
  const [inputUrl, setInputUrl] = createSignal<string | null>(null);
  const [showBoxes, setShowBoxes] = createSignal(true);

  let timer: number | undefined;

  void fetchHealth()
    .then(setHealth)
    .catch(() => setHealth(null));

  onCleanup(() => {
    if (timer) window.clearInterval(timer);
    if (inputUrl()) URL.revokeObjectURL(inputUrl()!);
  });

  const chooseFile = (next: File) => {
    if (inputUrl()) URL.revokeObjectURL(inputUrl()!);
    setInputUrl(next.type.startsWith("image/") ? URL.createObjectURL(next) : null);
    setFile(next);
    setResult(null);
    setError(null);
    setPhase("idle");
  };

  const clearFile = () => {
    if (inputUrl()) URL.revokeObjectURL(inputUrl()!);
    setInputUrl(null);
    setFile(null);
    setResult(null);
    setError(null);
    setPhase("idle");
  };

  const pickExample = async (path: string, name: string) => {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error("could not load the example");
      const blob = await res.blob();
      chooseFile(new File([blob], name, { type: blob.type || "image/jpeg" }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not load the example");
      setPhase("error");
    }
  };

  const start = async () => {
    const f = file();
    if (!f || phase() === "running") return;
    setPhase("running");
    setError(null);
    setElapsed(0);
    const startedAt = performance.now();
    timer = window.setInterval(() => setElapsed((performance.now() - startedAt) / 1000), 100);
    try {
      const payload = await runRestore(f, {
        lang: lang(),
        deskew: deskew(),
        ocr: ocr(),
        allPages: allPages(),
        autoRotate: autoRotate(),
        outputs: {
          overlay: wantOverlay(),
          pdf: wantPdf(),
          txt: wantTxt(),
          md: wantMd(),
        },
      });
      setResult(payload);
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "The run failed.");
      setPhase("error");
    } finally {
      if (timer) window.clearInterval(timer);
    }
  };

  return {
    file,
    lang,
    setLang,
    deskew,
    setDeskew,
    ocr,
    setOcr,
    allPages,
    setAllPages,
    autoRotate,
    setAutoRotate,
    wantOverlay,
    setWantOverlay,
    wantPdf,
    setWantPdf,
    wantTxt,
    setWantTxt,
    wantMd,
    setWantMd,
    advancedOpen,
    setAdvancedOpen,
    phase,
    result,
    error,
    elapsed,
    health,
    inputUrl,
    showBoxes,
    setShowBoxes,
    chooseFile,
    clearFile,
    pickExample,
    start,
  };
}

export type StudioState = ReturnType<typeof createStudioState>;

export const EXAMPLE_FILES = [
  { label: "Invoice photo", path: "/examples/invoice-photo.png", name: "invoice-photo.png" },
  { label: "1955 letterpress", path: "/examples/letterpress-input.jpg", name: "letterpress.jpg" },
  { label: "Clean invoice", path: "/examples/clean-invoice.png", name: "clean-invoice.png" },
];

export function limitsLine(health: Health | null): string {
  if (!health) return "≤ 12 MB · first page of PDFs · 60-min retention";
  return `≤ ${health.limits.max_upload_mb} MB · up to ${health.limits.max_pages} PDF pages · ${health.limits.ttl_minutes}-min retention`;
}
