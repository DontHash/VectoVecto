#!/bin/bash
# vm_startup.sh — GCP VM bootstrap for Tier-C SR training (spot L4 safe).
# - installs nothing destructive, caches data on the persistent boot disk
# - resumes from the last GCS checkpoint automatically
# - syncs checkpoints/logs to GCS during training (train_v2.py does the ckpts)
set -x
exec > /var/log/sr_startup.log 2>&1

BUCKET="${BUCKET:-gs://theproject-sr-artifacts}"
WORK=/opt/srwork
DATA=/opt/srdata
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-340}"
ITERS="${ITERS:-30000}"
BATCH="${BATCH:-8}"
PATCH="${PATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "$WORK" "$DATA"
cd "$WORK" || exit 1

# ---- code ----
gcloud storage cp "$BUCKET/code/sr_code.tar.gz" /tmp/sr_code.tar.gz \
  && tar xzf /tmp/sr_code.tar.gz -C "$WORK"

# ---- weights ----
mkdir -p "$WORK/weights"
gcloud storage cp "$BUCKET/weights/RealESRGAN_x4plus.pth" "$WORK/weights/" 2>/dev/null || true
gcloud storage cp "$BUCKET/weights/realesr-general-x4v3.pth" "$WORK/weights/" 2>/dev/null || true

# ---- datasets (cached on boot disk; survives stop/start) ----
if [ ! -d "$DATA/DIV2K_train_HR" ]; then
  echo "downloading DIV2K train..."
  curl -sL -o "$DATA/div2k_train.zip" https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  python3 -c "import zipfile; zipfile.ZipFile('$DATA/div2k_train.zip').extractall('$DATA')"
  rm -f "$DATA/div2k_train.zip"
fi
if [ ! -d "$DATA/DIV2K_valid_HR" ]; then
  echo "downloading DIV2K valid..."
  curl -sL -o "$DATA/div2k_valid.zip" https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
  python3 -c "import zipfile; zipfile.ZipFile('$DATA/div2k_valid.zip').extractall('$DATA')"
  rm -f "$DATA/div2k_valid.zip"
fi

# ---- python deps (torch is preinstalled in the DLVM image) ----
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3 4 5; do
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 && break
  echo "apt attempt $i failed; retrying in 15s"
  sleep 15
done
python3 -m pip install -q --no-input scikit-image pyiqa opencv-python-headless 2>/dev/null || true

# ---- resume from GCS ----
mkdir -p "$WORK/artifacts/tier_c"
gcloud storage cp "$BUCKET/tier_c/latest.pth" "$WORK/artifacts/tier_c/latest.pth" 2>/dev/null || true

# ---- log sync loop ----
(
  while true; do
    sleep 300
    [ -f "$WORK/train.log" ] && gcloud storage cp "$WORK/train.log" "$BUCKET/tier_c/train.log" >/dev/null 2>&1
  done
) &

# ---- train ----
cd "$WORK" || exit 1
nohup python3 -u train_v2.py \
  --hr-dirs "$DATA/DIV2K_train_HR" \
  --val-hr-dir "$DATA/DIV2K_valid_HR" \
  --base-weights "$WORK/weights/RealESRGAN_x4plus.pth" \
  --out-dir "$WORK/artifacts/tier_c" \
  --arch rrdbnet --scale 4 \
  --iters "$ITERS" --batch "$BATCH" --patch "$PATCH" --amp bf16 \
  --num-workers "$NUM_WORKERS" \
  --val-every 1000 --val-images 8 --val-size 512 --val-severity medium \
  --save-every 500 --save-samples-every 2000 \
  --gcs-bucket "$BUCKET/tier_c" \
  --time-budget-min "$TIME_BUDGET_MIN" --shutdown-on-finish \
  >> "$WORK/train.log" 2>&1 &

echo "training launched"
