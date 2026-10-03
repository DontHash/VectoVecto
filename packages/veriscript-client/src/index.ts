/**
 * veriscript-client — typed API client for a VeriScript server.
 *
 * The OCR pipeline is Python (`webapp/server.py`, or the hosted demo); this
 * package is the JS/TS front door to it: upload a page, read the review
 * queue, apply corrections cumulatively, build the privacy-first training
 * export, and check/clear the optional local correction memory.
 *
 * Zero runtime dependencies; Node >= 18 or any modern browser
 * (`fetch` + `FormData` + `Blob`).
 */
import type {
  ClearMemoryResponse,
  CorrectResponse,
  CorrectionInput,
  ExportResponse,
  Health,
  Lang,
  MemoryStatus,
  OcrEngine,
  RunPayload,
} from "./types.js";

export * from "./types.js";

export interface ClientOptions {
  /** Server origin, e.g. `"https://veriscript.live"`. Empty = same origin. */
  baseUrl?: string;
  /** HTTP Basic credentials, when the server sets `WEB_USER`/`WEB_PASSWORD`. */
  auth?: { username: string; password: string };
  /** Custom fetch (undici agents, retries, tests). Defaults to the global. */
  fetch?: typeof fetch;
  /** Extra headers on every request. */
  headers?: Record<string, string>;
}

export class VeriScriptError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** Seconds from `Retry-After` (429 responses), when present. */
    readonly retryAfter?: number,
  ) {
    super(message);
    this.name = "VeriScriptError";
  }
}

export interface Outputs {
  overlay: boolean;
  pdf: boolean;
  txt: boolean;
  md: boolean;
}

export interface RestoreOptions {
  /** OCR language: `"ne"` (default, matches the studio), `"hi"`, `"en"`. */
  lang?: Lang;
  /** Deskew small rotations. */
  deskew?: boolean;
  /** OCR engine: `"auto"` (default), `"rapidocr"`, `"tesseract"`. */
  ocr?: OcrEngine;
  /** `false` (default): first page of a PDF only. */
  allPages?: boolean;
  /** `true` (default). */
  autoRotate?: boolean;
  /** Which artifacts to write; all four default on. */
  outputs?: Partial<Outputs>;
  /** Upload filename when `file` is a plain Blob (default `page.png`). */
  filename?: string;
  /** Cancel the upload. */
  signal?: AbortSignal;
}

async function toError(res: Response): Promise<VeriScriptError> {
  let message = `Request failed (${res.status})`;
  try {
    const data = (await res.json()) as { detail?: unknown; error?: unknown };
    const detail = data?.detail ?? data?.error;
    if (typeof detail === "string" && detail) message = detail;
  } catch {
    /* keep the default */
  }
  const retry = Number(res.headers.get("retry-after") || 0) || undefined;
  return new VeriScriptError(message, res.status, retry);
}

function encodeBasic(username: string, password: string): string {
  // HTTP Basic is ASCII by convention; btoa is global in Node >= 16 and browsers.
  return "Basic " + btoa(`${username}:${password}`);
}

export function createClient(options: ClientOptions = {}) {
  const baseUrl = (options.baseUrl ?? "").replace(/\/+$/, "");
  const fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  const baseHeaders: Record<string, string> = { ...(options.headers ?? {}) };
  if (options.auth) {
    baseHeaders.Authorization = encodeBasic(
      options.auth.username,
      options.auth.password,
    );
  }

  const url = (path: string) => `${baseUrl}${path}`;

  async function request(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = {
      ...baseHeaders,
      ...((init.headers as Record<string, string> | undefined) ?? {}),
    };
    const res = await fetchImpl(url(path), { ...init, headers });
    if (!res.ok) throw await toError(res);
    return res;
  }

  async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
    return (await request(path, init)).json() as Promise<T>;
  }

  return {
    /** The normalized base URL this client talks to. */
    baseUrl,

    /** Service status: version, busy flag, quota limits. */
    health: () => json<Health>("/api/health"),

    /**
     * Run the document pipeline on one page (image or first PDF page).
     * The call resolves when processing finishes; server limits (429 with
     * `Retry-After`) surface as `VeriScriptError`.
     */
    restore: async (file: Blob, opts: RestoreOptions = {}): Promise<RunPayload> => {
      const outputs: Outputs = {
        overlay: true,
        pdf: true,
        txt: true,
        md: true,
        ...opts.outputs,
      };
      const form = new FormData();
      const filename =
        opts.filename ??
        (typeof File !== "undefined" && file instanceof File
          ? file.name
          : "page.png");
      form.append("file", file, filename);
      form.append("lang", opts.lang ?? "ne");
      form.append("deskew", opts.deskew ? "1" : "0");
      form.append("ocr", opts.ocr ?? "auto");
      form.append("all_pages", opts.allPages ? "1" : "0");
      form.append("auto_rotate", opts.autoRotate === false ? "0" : "1");
      form.append("overlay", outputs.overlay ? "1" : "0");
      form.append("pdf", outputs.pdf ? "1" : "0");
      form.append("txt", outputs.txt ? "1" : "0");
      form.append("md", outputs.md ? "1" : "0");
      return json<RunPayload>("/api/restore", {
        method: "POST",
        body: form,
        signal: opts.signal,
      });
    },

    /**
     * Apply corrections (cumulative on the server). Post only what is new:
     * already-applied entries are idempotent but re-counted in `stats`.
     */
    correct: (runId: string, corrections: CorrectionInput[]) =>
      json<CorrectResponse>(`/api/runs/${runId}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corrections }),
      }),

    /**
     * Build the privacy-first training archive. Only `share` indices are
     * cropped and packed; nothing is uploaded anywhere.
     */
    exportCorrections: (
      runId: string,
      corrections: CorrectionInput[],
      share: number[],
    ) =>
      json<ExportResponse>(`/api/runs/${runId}/corrections/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corrections, share }),
      }),

    /** Local correction memory status (off by default on hosted servers). */
    memory: () => json<MemoryStatus>("/api/memory"),

    /** Clear the local correction memory. */
    clearMemory: () =>
      json<ClearMemoryResponse>("/api/memory/clear", { method: "POST" }),

    /** Absolute URL of a run artifact (e.g. `"corrected.txt"`). */
    fileUrl: (runId: string, name: string) =>
      url(`/api/runs/${runId}/files/${name}`),

    /** Fetch a run artifact as a raw `Response` (streams, no buffering). */
    downloadFile: (runId: string, name: string) =>
      request(`/api/runs/${runId}/files/${name}`),
  };
}

export type VeriScriptClient = ReturnType<typeof createClient>;
