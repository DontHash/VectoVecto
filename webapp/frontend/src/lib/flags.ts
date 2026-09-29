/** Flag semantics — mirrors document_export.FLAG_COLORS.
 *
 *  red   → digit_conflict (the alternative reading is kept)
 *  amber → low_conf · digit_uncertain · unknown_word
 *  neutral → informational (invalid_sequence, script_mismatch); these render
 *            green in the overlay PNG but sit in the queue as notes.
 */
import type { FlagName } from "./types";

export type FlagTone = "green" | "amber" | "red" | "neutral";

const FLAG_META: Record<string, { tone: FlagTone; label: string }> = {
  digit_conflict: { tone: "red", label: "digit conflict" },
  digit_uncertain: { tone: "amber", label: "digit uncertain" },
  low_conf: { tone: "amber", label: "low confidence" },
  unknown_word: { tone: "amber", label: "unknown word" },
  invalid_sequence: { tone: "neutral", label: "invalid sequence" },
  script_mismatch: { tone: "neutral", label: "script mismatch" },
};

export function flagMeta(flag: FlagName): { tone: FlagTone; label: string } {
  return FLAG_META[flag] ?? { tone: "neutral", label: String(flag).replace(/_/g, " ") };
}

/** Overall tone of a queue row: the worst flag wins. */
export function rowTone(flags: FlagName[]): FlagTone {
  if (flags.includes("digit_conflict")) return "red";
  if (
    flags.includes("digit_uncertain") ||
    flags.includes("low_conf") ||
    flags.includes("unknown_word")
  ) {
    return "amber";
  }
  return "neutral";
}

/** Risk bars are scaled to the queue's own maximum (7.0 in current runs). */
export function riskTone(risk: number | null): "high" | "mid" | "low" {
  if (risk === null) return "low";
  if (risk >= 5) return "high";
  if (risk > 1) return "mid";
  return "low";
}
