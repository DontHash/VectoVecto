# pull.ps1 — sync training artifacts (checkpoints, logs, val samples) from GCS.
#
# Usage:
#   powershell -File gcp\pull.ps1
#   powershell -File gcp\pull.ps1 -DeleteVm   # stop paying: delete the VM (disk dies too)
param(
  [string]$Project = "theproject-sr",
  [string]$Zone = "us-central1-a",
  [string]$Instance = "sr-train",
  [string]$Bucket = "gs://theproject-sr-artifacts",
  [switch]$DeleteVm,
  [switch]$StopVm
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
$dest = Join-Path $root "artifacts\tier_c"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Write-Host "[pull] syncing $Bucket/tier_c -> $dest"
gcloud storage rsync "$Bucket/tier_c" $dest --recursive --project $Project

if ($StopVm) {
  Write-Host "[pull] stopping instance $Instance (boot disk persists, billing stops)"
  gcloud compute instances stop $Instance --project $Project --zone $Zone
}
if ($DeleteVm) {
  Write-Host "[pull] deleting instance $Instance (this deletes the cached dataset disk)"
  gcloud compute instances delete $Instance --project $Project --zone $Zone --quiet
}
Write-Host "[pull] done."
