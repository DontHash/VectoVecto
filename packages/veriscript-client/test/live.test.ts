/**
 * Optional integration test against a real server.
 *
 *   VERISCRIPT_BASE_URL=http://127.0.0.1:8000 npm test
 *
 * Skipped otherwise, so `npm test` stays offline.
 */
import { describe, expect, it } from "vitest";

import { createClient } from "../src/index.js";

const baseUrl = process.env.VERISCRIPT_BASE_URL;

describe.skipIf(!baseUrl)("live server", () => {
  it("health answers", async () => {
    const client = createClient({ baseUrl: baseUrl! });
    const health = await client.health();
    expect(health.ok).toBe(true);
    expect(typeof health.version).toBe("string");
  });
});
