#!/usr/bin/env bash
# gcp_w1_vm_startup.sh — train the W1 Devanagari CRNN on a Deep Learning VM (W-C).
#
# Metadata: WHEEL_URI (gs://.../deva_crnn-*.whl), OUT_URI (gs:// prefix for
# per-epoch checkpoints), NPZ_NAME (data file inside the wheel). The DLVM
# image already carries torch; the wheel is installed --no-deps. Checkpoints
# and metrics upload to GCS every epoch (`_upload_if_gcs`), then the instance
# stops itself. Log: /opt/w1/train.log.
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

WHEEL_URI="$(meta WHEEL_URI)"
OUT_URI="$(meta OUT_URI)"
NPZ_NAME="$(meta NPZ_NAME)"
EPOCHS="$(meta EPOCHS || echo 26)"
BATCH="$(meta BATCH || echo 64)"
HIDDEN="$(meta HIDDEN || echo 256)"
LR_SCHEDULE="$(meta LR_SCHEDULE || echo none)"

# DLVM images ship the *open* kernel modules, which do not support V100
# (Volta): nvidia-smi fails and torch falls back to CPU. Install the
# proprietary driver and reboot once; the startup script re-runs on boot.
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[w1] NVIDIA driver not ready; installing nvidia-driver-580 (proprietary)"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver-580
  echo "[w1] rebooting so the driver loads; the startup script re-runs on boot"
  reboot
  exit 0
fi

mkdir -p /opt/w1
exec > >(tee -a /opt/w1/train.log) 2>&1
echo "[w1] $(date -u) wheel=$WHEEL_URI out=$OUT_URI npz=$NPZ_NAME"

gcloud storage cp "$WHEEL_URI" /opt/w1/
WHEEL="/opt/w1/$(basename "$WHEEL_URI")"
python3 -m pip install --no-deps --quiet "$WHEEL"
python3 -c "import deva_crnn, torch; print('[w1] deva_crnn', deva_crnn.__file__, 'cuda', torch.cuda.is_available())"

python3 -m deva_crnn.train \
  --data "pkg://data/$NPZ_NAME" \
  --out "$OUT_URI" \
  --epochs "$EPOCHS" --batch "$BATCH" --lr 1e-3 \
  --in-h 48 --in-w 512 --hidden "$HIDDEN" --lr-schedule "$LR_SCHEDULE" \
  --max-hours 3.0 --workers 2 --save-best

echo "[w1] $(date -u) training finished; stopping instance"
shutdown -h now
