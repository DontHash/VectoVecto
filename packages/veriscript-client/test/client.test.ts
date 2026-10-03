import { describe, expect, it, vi } from "vitest";

import { createClient, VeriScriptError, type RunPayload } from "../src/index.js";

function json(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

const runPayload = {
  run_id: "abc123",
  meta: {},
  flags_summary: {},
  review: [],
  transcript: "",
  status_line: "",
  files: { restored: "/x" },
  expires_in: 60,
} as unknown as RunPayload;

describe("createClient", () => {
  it("calls health with the base URL and basic auth", async () => {
    const fetchMock = vi.fn(
      async (_url: string, _init?: RequestInit) =>
        json({ ok: true, version: "1.5.0", busy: false, limits: {} }),
    );
    const client = createClient({
      baseUrl: "http://cv.local:8000/",
      auth: { username: "u", password: "p" },
      fetch: fetchMock as unknown as typeof fetch,
    });
    const health = await client.health();
    expect(health.ok).toBe(true);
    const [calledUrl, init] = fetchMock.mock.calls[0]!;
    expect(calledUrl).toBe("http://cv.local:8000/api/health");
    expect((init?.headers as Record<string, string>).Authorization).toBe(
      "Basic " + btoa("u:p"),
    );
  });

  it("uploads with the studio's form fields and defaults", async () => {
    let captured: RequestInit | undefined;
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      captured = init;
      return json(runPayload);
    });
    const client = createClient({
      baseUrl: "http://x",
      fetch: fetchMock as unknown as typeof fetch,
    });
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });
    const out = await client.restore(blob, {
      filename: "page.png",
      lang: "en",
      outputs: { md: false },
    });
    expect(out.run_id).toBe("abc123");
    const form = captured?.body as FormData;
    expect(form.get("lang")).toBe("en");
    expect(form.get("ocr")).toBe("auto");
    expect(form.get("auto_rotate")).toBe("1");
    expect(form.get("pdf")).toBe("1");
    expect(form.get("md")).toBe("0");
    expect((form.get("file") as File).name).toBe("page.png");
    expect(fetchMock.mock.calls[0]![0]).toBe("http://x/api/restore");
  });

  it("posts corrections as JSON", async () => {
    let captured: RequestInit | undefined;
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      captured = init;
      return json({ run_id: "r1", stats: {}, flags_summary: {}, review: [], transcript: "", files: {} });
    });
    const client = createClient({
      fetch: fetchMock as unknown as typeof fetch,
    });
    const correction = {
      index: 0,
      bbox: [0, 0, 1, 1] as [number, number, number, number],
      original: "A",
      corrected: "B",
      action: "changed" as const,
    };
    await client.correct("r1", [correction]);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/runs/r1/correct");
    expect(JSON.parse(captured?.body as string)).toEqual({
      corrections: [correction],
    });
  });

  it("maps HTTP errors to VeriScriptError with Retry-After", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: "free demo limit" }), {
          status: 429,
          headers: { "Retry-After": "42" },
        }),
    );
    const client = createClient({
      fetch: fetchMock as unknown as typeof fetch,
    });
    const error = await client.restore(new Blob(["x"])).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(VeriScriptError);
    const err = error as VeriScriptError;
    expect(err.status).toBe(429);
    expect(err.retryAfter).toBe(42);
    expect(err.message).toBe("free demo limit");
  });

  it("builds artifact URLs and surfaces missing files", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      url.endsWith("corrected.txt")
        ? new Response("hi", { status: 200 })
        : json({ detail: "This result has expired." }, { status: 404 }),
    );
    const client = createClient({
      baseUrl: "https://veriscript.live",
      fetch: fetchMock as unknown as typeof fetch,
    });
    expect(client.fileUrl("r1", "corrected.txt")).toBe(
      "https://veriscript.live/api/runs/r1/files/corrected.txt",
    );
    const res = await client.downloadFile("r1", "corrected.txt");
    expect(await res.text()).toBe("hi");
    const missing = await client
      .downloadFile("r1", "missing.txt")
      .catch((e: unknown) => e);
    expect(missing).toBeInstanceOf(VeriScriptError);
  });
});
