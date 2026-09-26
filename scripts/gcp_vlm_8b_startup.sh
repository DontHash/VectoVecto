#!/usr/bin/env bash
# gcp_vlm_8b_startup.sh — Qwen3-VL-8B fp16 bake-off on the SPOT V100 (Phase 1).
#
# 8B fp16 (~16.4 GB) exceeds the 16 GB V100, so `device_map="auto"` offloads
# the remainder to the 14 GB host RAM (with ~4 GB of VRAM reserved for the
# vision activations of a full page). Pages run first (the decisive number),
# then lines. Results upload to $BUCKET_URI/results/ and the instance stops.
#
# Metadata: BUCKET_URI (gs:// prefix holding code.tar.gz + data.tar.gz),
# optional RUN_MODE = all (default) | pages | lines.
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

BUCKET_URI="$(meta BUCKET_URI)"
RUN_MODE="$(meta RUN_MODE || echo all)"

# DLVM open kernel modules do not support V100: install the proprietary
# driver and reboot once (the startup script re-runs on boot).
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[vlm8b] NVIDIA driver not ready; installing nvidia-driver-580"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver-580
  reboot
  exit 0
fi

mkdir -p /opt/vlm
cd /opt/vlm
exec > >(tee -a /opt/vlm/eval8b.log) 2>&1
echo "[vlm8b] $(date -u) start bucket=$BUCKET_URI"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2
df -h / | tail -1

gcloud storage cp "$BUCKET_URI/code.tar.gz" /opt/vlm/
gcloud storage cp "$BUCKET_URI/data.tar.gz" /opt/vlm/
tar xzf code.tar.gz
tar xzf data.tar.gz

python3 -m pip install --quiet --upgrade transformers accelerate \
  "opencv-python-headless" jiwer Pillow "jinja2>=3.1" rapidocr
export HF_HOME=/opt/vlm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

# Diagnostics: the manifest fix must make all pages readable on POSIX.
python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/vlm")
import doc_data
m = doc_data.load_dataset("data/doc_eval/nepali_pdf_v2")
bad = [e["id"] for e in m["entries"]
       if doc_data.imread_safe(e["_degraded_path"]) is None]
print("[vlm8b] entries:", len(m["entries"]), "| unreadable:", len(bad))
print("[vlm8b] first path:", m["entries"][0]["_degraded_path"])
PY

if [ "$RUN_MODE" = "all" ] || [ "$RUN_MODE" = "pages" ]; then
  python3 evals/harness/eval_models.py --model qwen3vl --mode pages \
    --data-dir data/doc_eval/nepali_pdf_v2 --limit 10 \
    --json out/qwen8b_pages.json || echo "[vlm8b] PAGES FAILED"
fi
if [ "$RUN_MODE" = "all" ] || [ "$RUN_MODE" = "lines" ]; then
  python3 evals/harness/eval_models.py --model qwen3vl --mode lines \
    --lines-source deva_real_lines --limit 150 \
    --json out/qwen8b_lines.json || echo "[vlm8b] LINES FAILED"
fi

for f in out/qwen8b_pages.json out/qwen8b_lines.json; do
  [ -f "$f" ] && gcloud storage cp "$f" "$BUCKET_URI/results/" || true
done
gcloud storage cp /opt/vlm/eval8b.log "$BUCKET_URI/results/eval8b.log" || true
echo "[vlm8b] $(date -u) done; stopping instance"
shutdown -h now
