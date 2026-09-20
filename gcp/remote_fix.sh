#!/bin/bash
# remote_fix.sh — repair libGL on the DLVM image (boot-time dpkg lock can make
# the startup-script apt step fail silently) and launch Tier-C training.
set -x
export DEBIAN_FRONTEND=noninteractive

for i in 1 2 3 4 5; do
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 && break
  echo "apt attempt $i failed; retrying in 15s"
  sleep 15
done

python3 -c 'import cv2; print("cv2 OK", cv2.__version__)' || exit 1

if pgrep -f train_v2.py >/dev/null; then
  echo "training already running:"
  pgrep -af train_v2.py | head -1
  exit 0
fi

mkdir -p /opt/srwork/artifacts/tier_c
cd /opt/srwork || exit 1
nohup python3 -u train_v2.py \
  --hr-dirs /opt/srdata/DIV2K_train_HR \
  --val-hr-dir /opt/srdata/DIV2K_valid_HR \
  --base-weights /opt/srwork/weights/RealESRGAN_x4plus.pth \
  --out-dir /opt/srwork/artifacts/tier_c \
  --arch rrdbnet --scale 4 \
  --iters 30000 --batch 8 --patch 64 --amp bf16 --num-workers 4 \
  --val-every 1000 --val-images 8 --val-size 512 --val-severity medium \
  --save-every 500 --save-samples-every 2000 \
  --gcs-bucket gs://theproject-sr-artifacts/tier_c \
  --time-budget-min 340 --shutdown-on-finish \
  >> /opt/srwork/train.log 2>&1 &
echo "launched pid $!"
sleep 25
tail -6 /opt/srwork/train.log
