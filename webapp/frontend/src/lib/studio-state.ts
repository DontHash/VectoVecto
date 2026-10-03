/** Studio state — one factory, used by the homepage hero and the full studio
 *  page, so the product behaves identically wherever it is embedded. */
import { createMemo, createSignal, onCleanup } from "solid-js";
import { clearMemory as clearMemoryApi, correctRun, exportRunCorrections, fetchHealth, runRestore } from "./api";
import type {
  CorrectionInput,
  CorrectionStats,
  ExportResponse,
  Health,
  Lang,
  OcrEngine,
  RunPayload,
} from "./types";

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
  const [corrections, setCorrections] = createSignal<CorrectionInput[]>([]);
  const [correcting, setCorrecting] = createSignal(false);
  const [correctionError, setCorrectionError] = createSignal<string | null>(null);
  const [lastStats, setLastStats] = createSignal<CorrectionStats | null>(null);
  const [exportResult, setExportResult] = createSignal<ExportResponse | null>(null);
  const [memoryCleared, setMemoryCleared] = createSignal<number | null>(null);
  // Signature of every correction that has already been applied. The batch is
  // kept after applying (the export panel and the next POST need it), so the
  // pending count must ignore what was already sent — otherwise "apply" stays
  // enabled and re-posts the same corrections.
  const [appliedSigs, setAppliedSigs] = createSignal<Record<number, string>>({});

  const correctionSig = (c: CorrectionInput) =>
    [c.index, c.corrected, c.action, c.suggested ?? "", c.suggestion_source ?? ""].join("\u0000");

  const pendingCorrections = createMemo(() =>
    corrections().filter((c) => appliedSigs()[c.index] !== correctionSig(c)),
  );

  const resetCorrections = () => {
    setCorrections([]);
    setAppliedSigs({});
    setCorrectionError(null);
    setLastStats(null);
    setExportResult(null);
  };

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
    resetCorrections();
  };

  const clearFile = () => {
    if (inputUrl()) URL.revokeObjectURL(inputUrl()!);
    setInputUrl(null);
    setFile(null);
    setResult(null);
    setError(null);
    setPhase("idle");
    resetCorrections();
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
      resetCorrections();
    } catch (e) {
      setError(e instanceof Error ? e.message : "The run failed.");
      setPhase("error");
    } finally {
      if (timer) window.clearInterval(timer);
    }
  };

  const stageCorrection = (correction: CorrectionInput) => {
    setCorrectionError(null);
    setCorrections((prev) => [
      ...prev.filter((c) => c.index !== correction.index),
      correction,
    ]);
  };

  const unstageCorrection = (index: number) => {
    setCorrections((prev) => prev.filter((c) => c.index !== index));
  };

  const correctionFor = (index: number) =>
    corrections().find((c) => c.index === index);

  const applyCorrections = async () => {
    const p = result();
    // Post only what is new; the server applies it on top of the run's
    // current corrected state (cumulative), so already-applied corrections
    // are neither re-counted nor re-recorded.
    const list = pendingCorrections();
    if (!p || !list.length || correcting()) return;
    setCorrecting(true);
    setCorrectionError(null);
    try {
      const res = await correctRun(p.run_id, list);
      setResult({
        ...p,
        review: res.review,
        flags_summary: res.flags_summary,
        transcript: res.transcript,
        files: { ...p.files, ...res.files },
      });
      setLastStats(res.stats);
      setExportResult(null);
      setAppliedSigs((prev) => ({
        ...prev,
        ...Object.fromEntries(list.map((c) => [c.index, correctionSig(c)])),
      }));
    } catch (e) {
      setCorrectionError(e instanceof Error ? e.message : "Could not apply the corrections.");
    } finally {
      setCorrecting(false);
    }
  };

  const exportCorrections = async (share: number[]) => {
    const p = result();
    const list = corrections();
    if (!p || !list.length || correcting()) return;
    setCorrecting(true);
    setCorrectionError(null);
    try {
      setExportResult(await exportRunCorrections(p.run_id, list, share));
    } catch (e) {
      setCorrectionError(
        e instanceof Error ? e.message : "Could not build the corrections archive.",
      );
    } finally {
      setCorrecting(false);
    }
  };

  const clearMemory = async () => {
    setCorrectionError(null);
    try {
      const out = await clearMemoryApi();
      setMemoryCleared(out.removed);
    } catch (e) {
      setCorrectionError(
        e instanceof Error ? e.message : "Could not clear the local memory.",
      );
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
    corrections,
    pendingCorrections,
    correcting,
    correctionError,
    lastStats,
    exportResult,
    stageCorrection,
    unstageCorrection,
    correctionFor,
    applyCorrections,
    exportCorrections,
    memoryCleared,
    clearMemory,
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
