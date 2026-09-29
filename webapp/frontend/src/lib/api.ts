/** Typed API client. Same-origin; the dev server proxies /api to :8000. */
import type { Health, Lang, OcrEngine, RunPayload } from "./types";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfter?: number,
  ) {
    super(message);
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  try {
    const data = (await res.json()) as { error?: string };
    if (data?.error) message = data.error;
  } catch {
    /* keep the default */
  }
  const retry = Number(res.headers.get("retry-after") || 0) || undefined;
  return new ApiError(message, res.status, retry);
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch("/api/health");
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as Health;
}

export interface RestoreOptions {
  lang: Lang;
  deskew: boolean;
  ocr: OcrEngine;
  allPages: boolean;
  autoRotate: boolean;
  outputs: { overlay: boolean; pdf: boolean; txt: boolean; md: boolean };
}

export async function runRestore(
  file: File | Blob,
  opts: RestoreOptions,
): Promise<RunPayload> {
  const body = new FormData();
  const name = file instanceof File ? file.name : "example.png";
  body.append("file", file, name);
  body.append("lang", opts.lang);
  body.append("deskew", opts.deskew ? "1" : "0");
  body.append("ocr", opts.ocr);
  body.append("all_pages", opts.allPages ? "1" : "0");
  body.append("auto_rotate", opts.autoRotate ? "1" : "0");
  body.append("overlay", opts.outputs.overlay ? "1" : "0");
  body.append("pdf", opts.outputs.pdf ? "1" : "0");
  body.append("txt", opts.outputs.txt ? "1" : "0");
  body.append("md", opts.outputs.md ? "1" : "0");

  const res = await fetch("/api/restore", { method: "POST", body });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as RunPayload;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export { ApiError };
