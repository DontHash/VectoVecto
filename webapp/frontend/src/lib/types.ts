/** API + domain types (mirrors webapp/vvweb payloads). */

export type FlagName =
  | "digit_conflict"
  | "digit_uncertain"
  | "low_conf"
  | "unknown_word"
  | "invalid_sequence"
  | "script_mismatch"
  | (string & {});

export interface QueueItem {
  text: string;
  alt_text: string | null;
  flags: FlagName[];
  conf: number | null;
  risk: number | null;
  bbox: [number, number, number, number] | null;
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
}

export interface RunPayload {
  run_id: string;
  meta: RunMeta;
  flags_summary: Record<string, number>;
  review: QueueItem[];
  transcript: string;
  status_line: string;
  files: {
    restored: string;
    overlay: string;
    pdf: string;
    txt: string;
    md: string;
    json: string;
  };
  expires_in: number;
}

export interface Health {
  ok: boolean;
  version: string;
  busy: boolean;
  limits: {
    max_upload_mb: number;
    max_image_megapixels: number;
    ttl_minutes: number;
    rate_max: number;
    rate_window_s: number;
    auth: boolean;
  };
}

export type Lang = "ne" | "hi" | "en";
