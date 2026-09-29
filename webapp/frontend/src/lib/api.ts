/** Typed API client. Same-origin; the dev server proxies /api to :8000. */
import type { Health, Lang, RunPayload } from "./types";

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

export async function runRestore(
  file: File | Blob,
  opts: { lang: Lang; deskew: boolean },
): Promise<RunPayload> {
  const body = new FormData();
  const name = file instanceof File ? file.name : "example.png";
  body.append("file", file, name);
  body.append("lang", opts.lang);
  body.append("deskew", opts.deskew ? "1" : "0");

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
