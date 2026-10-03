/** API + domain types (mirrors webapp/vvweb payloads). */

export type FlagName =
  | "digit_conflict"
  | "digit_uncertain"
  | "low_conf"
  | "unknown_word"
  | "invalid_sequence"
  | "script_mismatch"
  | (string & {});

/** One candidate label for a flagged token (docs/HARNESS_PLAN.md §2). */
export interface Suggestion {
  text: string;
  source: string;
  why?: string;
}

export interface QueueItem {
  index: number | null;
  text: string;
  alt_text: string | null;
  flags: FlagName[];
  conf: number | null;
  risk: number | null;
  bbox: [number, number, number, number] | null;
  /** Additive harness evidence; absent or empty on untouched runs. */
  suggestions?: Suggestion[];
}

export interface RunMeta {
  seconds: number;
  backend: string;
  primary_stream: string | null;
  skew_angle: number;
  resized: boolean;
  n_tokens: number;
  flagged: number;
  digit_conflicts: number;
  lang: string;
  input_kind: string;
  deskew: boolean;
  auto_rotate: boolean;
  all_pages: boolean;
  pages_processed: number;
  pages_capped: boolean;
  page_cap: number;
}

export interface RunFiles {
  restored: string;
  overlay?: string;
  pdf?: string;
  txt?: string;
  md?: string;
  json?: string;
  combined_pdf?: string;
  combined_txt?: string;
  combined_md?: string;
  corrected_pdf?: string;
  corrected_txt?: string;
  corrected_md?: string;
  corrected_json?: string;
  corrections_json?: string;
  corrections_zip?: string;
}

export interface RunPayload {
  run_id: string;
  meta: RunMeta;
  flags_summary: Record<string, number>;
  review: QueueItem[];
  transcript: string;
  status_line: string;
  files: RunFiles;
  expires_in: number;
}

/** docs/CORRECTIONS.md — one staged fix, keyed to a token index. */
export interface CorrectionInput {
  index: number;
  bbox: [number, number, number, number] | null;
  original: string;
  corrected: string;
  action: "changed" | "confirmed";
}

export interface CorrectionStats {
  changed: number;
  confirmed: number;
  skipped: number;
  skipped_indices: number[];
  reviewed: number;
}

export interface CorrectResponse {
  run_id: string;
  stats: CorrectionStats;
  flags_summary: Record<string, number>;
  review: QueueItem[];
  transcript: string;
  files: RunFiles;
}

export interface ExportResponse {
  run_id: string;
  file: string;
  shared: number;
}

export interface Health {
  ok: boolean;
  version: string;
  busy: boolean;
  limits: {
    max_upload_mb: number;
    max_image_megapixels: number;
    max_pages: number;
    ttl_minutes: number;
    rate_max: number;
    rate_window_s: number;
    auth: boolean;
  };
}

export type Lang = "ne" | "hi" | "en";
export type OcrEngine = "auto" | "rapidocr" | "tesseract";
