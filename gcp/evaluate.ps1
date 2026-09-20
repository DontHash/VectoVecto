# evaluate.ps1 — pull the fine-tuned checkpoints, then run the acceptance eval
# against the baselines on DIV2K-valid synthetic pairs.
#
# Usage:
#   powershell -File gcp\evaluate.ps1                       # medium severity, 20 imgs
#   powershell -File gcp\evaluate.ps1 -Severity heavy -Images 30
param(
  [string]$Project = "theproject-sr",
  [string]$Zone = "us-central1-a",
  [string]$Bucket = "gs://theproject-sr-artifacts",
  [string]$HrDir = "data\DIV2K_valid_HR",
  [int]$Images = 20,
  [string]$Severity = "medium",
  [string]$Model = "artifacts\tier_c\g_ema.pth"
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
Set-Location $root

Write-Host "[evaluate] syncing checkpoints from GCS..."
gcloud storage rsync "$Bucket/tier_c" "artifacts\tier_c" --recursive --project $Project

if (-not (Test-Path $Model)) {
  Write-Warning "checkpoint $Model not found after sync (training may not have saved one yet)"
  exit 1
}

Write-Host "[evaluate] running harness (severity=$Severity images=$Images)..."
python eval_harness_v2.py `
  --hr-dir $HrDir --images $Images --hr-size 512 --severity $Severity `
  --models "bicubic,lanczos,x4plus,x4v3,tierb,pth:$Model" `
  --json "out\acceptance_$Severity.json" `
  --csv "out\acceptance_$Severity.csv" `
  --crops "out\acceptance_crops_$Severity"
Write-Host "[evaluate] done. Reports in out\acceptance_$Severity.*"
