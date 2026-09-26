#!/usr/bin/env bash
# gcp_vlm_eval_startup.sh — Qwen3-VL bake-off on a Deep Learning VM (Phase 1).
#
# Metadata: BUCKET_URI (gs:// prefix holding code.tar.gz + data.tar.gz). The
# tars mirror the repo layout (evals/harness, doc_*.py, data/doc_eval). Results
# upload to $BUCKET_URI/results/ and the instance stops itself.
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

BUCKET_URI="$(meta BUCKET_URI)"

# DLVM open kernel modules do not support V100: install the proprietary
# driver and reboot once (the startup script re-runs on boot).
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[vlm] NVIDIA driver not ready; installing nvidia-driver-580"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver-580
  reboot
  exit 0
fi

mkdir -p /opt/vlm
cd /opt/vlm
exec > >(tee -a /opt/vlm/eval.log) 2>&1
echo "[vlm] $(date -u) start bucket=$BUCKET_URI"

gcloud storage cp "$BUCKET_URI/code.tar.gz" /opt/vlm/
gcloud storage cp "$BUCKET_URI/data.tar.gz" /opt/vlm/
tar xzf code.tar.gz
tar xzf data.tar.gz

python3 -m pip install --quiet --upgrade transformers accelerate \
  "opencv-python-headless" jiwer Pillow "jinja2>=3.1" rapidocr
export HF_HOME=/opt/vlm/hf
mkdir -p out

# Diagnostics: did the Devanagari page filenames survive extraction?
ls data/doc_eval/nepali_pdf_v2/pages | head -3
python3 - <<'PY'
import os, sys
sys.path.insert(0, "/opt/vlm")
import doc_data
m = doc_data.load_dataset("data/doc_eval/nepali_pdf_v2")
e = m["entries"][0]
print("[vlm] entries:", len(m["entries"]),
      "| first:", e["_degraded_path"], "| exists:",
      os.path.exists(e["_degraded_path"]))
img = doc_data.imread_safe(e["_degraded_path"])
print("[vlm] imread:", None if img is None else img.shape)
PY

python3 evals/harness/eval_models.py --model qwen3vl-4b --mode lines \
  --lines-source deva_real_lines --limit 150 --json out/qwen4b_lines.json \
  || echo "[vlm] LINES FAILED"
python3 evals/harness/eval_models.py --model qwen3vl-4b --mode pages \
  --data-dir data/doc_eval/nepali_pdf_v2 --limit 10 --json out/qwen4b_pages.json \
  || echo "[vlm] PAGES FAILED"

gcloud storage cp out/qwen4b_lines.json out/qwen4b_pages.json \
  "$BUCKET_URI/results/" || true
gcloud storage cp /opt/vlm/eval.log "$BUCKET_URI/results/eval.log" || true
echo "[vlm] $(date -u) done; stopping instance"
shutdown -h now
