# launch.ps1 — package code+weights, upload to GCS, create/start the training VM.
#
# Usage:
#   powershell -File gcp\launch.ps1                 # create (or start) and train
#   powershell -File gcp\launch.ps1 -Recreate       # delete + recreate instance
#
# Cost: g2-standard-4 SPOT in us-central1 ≈ $0.30/hr. The instance auto-stops
# when training finishes (--shutdown-on-finish) or on preemption.
param(
  [string]$Project = "theproject-sr",
  [string]$Zone = "us-central1-a",
  [string]$Instance = "sr-train",
  [string]$Bucket = "gs://theproject-sr-artifacts",
  [string]$MachineType = "g2-standard-4",
  [string]$ImageFamily = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580",
  [int]$TimeBudgetMin = 340,
  [int]$Iters = 30000,
  [switch]$Spot,
  [switch]$Recreate
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
  foreach ($c in @("$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin",
                   "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin",
                   "C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin")) {
    if (Test-Path (Join-Path $c "gcloud.cmd")) { $env:Path = "$c;$env:Path"; break }
  }
}
$root = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $env:TEMP "sr_stage"
$tar = Join-Path $env:TEMP "sr_code.tar.gz"

$codeFiles = @(
  "train_v2.py", "degradation_v2.py", "rrdbnet.py", "srvggnet.py",
  "unet_discriminator.py", "perceptual_loss.py", "eval_harness_v2.py",
  "drunet.py", "deep_unfolding.py", "smart_upscaler.py", "esrgan_inference.py"
)

Write-Host "[launch] staging code..."
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($f in $codeFiles) {
  $src = Join-Path $root $f
  if (Test-Path $src) { Copy-Item $src $stage -Force } else { Write-Warning "missing $f" }
}
if (Test-Path $tar) { Remove-Item $tar -Force }
tar -czf $tar -C $stage .

Write-Host "[launch] uploading code + weights to $Bucket ..."
gcloud storage cp $tar "$Bucket/code/sr_code.tar.gz" --project $Project | Out-Null
gcloud storage cp (Join-Path $root "weights\RealESRGAN_x4plus.pth") `
  "$Bucket/weights/RealESRGAN_x4plus.pth" --project $Project | Out-Null
if (Test-Path (Join-Path $root "weights\realesr-general-x4v3.pth")) {
  gcloud storage cp (Join-Path $root "weights\realesr-general-x4v3.pth") `
    "$Bucket/weights/realesr-general-x4v3.pth" --project $Project | Out-Null
}

$exists = $null
try {
  $exists = gcloud compute instances describe $Instance --project $Project --zone $Zone --format="value(name)" 2>$null
} catch { $exists = $null }

if ($Recreate -and $exists) {
  Write-Host "[launch] deleting existing instance $Instance ..."
  gcloud compute instances delete $Instance --project $Project --zone $Zone --quiet
  $exists = $null
}

# metadata: pass train sizing through startup script env
$meta = "BUCKET=$Bucket,TIME_BUDGET_MIN=$TimeBudgetMin,ITERS=$Iters"

if ($exists) {
  Write-Host "[launch] instance exists -> starting it (startup re-runs, resumes from GCS)"
  gcloud compute instances start $Instance --project $Project --zone $Zone
} else {
  $prov = if ($Spot) { "SPOT" } else { "STANDARD" }
  Write-Host "[launch] creating $prov instance $Instance ..."
  $createArgs = @(
    "compute", "instances", "create", $Instance,
    "--project=$Project", "--zone=$Zone",
    "--machine-type=$MachineType",
    "--provisioning-model=$prov",
    "--maintenance-policy=TERMINATE",
    "--image-family=$ImageFamily", "--image-project=deeplearning-platform-release",
    "--boot-disk-size=150GB", "--boot-disk-type=pd-balanced",
    "--scopes=cloud-platform",
    "--metadata=$meta",
    "--metadata-from-file=startup-script=$PSScriptRoot\vm_startup.sh"
  )
  if ($Spot) { $createArgs += "--instance-termination-action=STOP" }
  gcloud @createArgs
}

Write-Host "[launch] done. Follow training:"
Write-Host "  gcloud compute ssh $Instance --project $Project --zone $Zone --command 'tail -f /opt/srwork/train.log'"
Write-Host "  powershell -File gcp\pull.ps1    # sync checkpoints + logs locally"
