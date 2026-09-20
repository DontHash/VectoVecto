# status.ps1 — one-glance status of the Tier-C training VM.
#
# Usage: powershell -File gcp\status.ps1
param(
  [string]$Project = "theproject-sr",
  [string]$Zone = "us-central1-a",
  [string]$Instance = "sr-train"
)
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
  foreach ($c in @("$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin",
                   "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin",
                   "C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin")) {
    if (Test-Path (Join-Path $c "gcloud.cmd")) { $env:Path = "$c;$env:Path"; break }
  }
}
$state = gcloud compute instances describe $Instance --project $Project --zone $Zone --format="value(status)" 2>$null
Write-Host "instance: $Instance  status: $state"
if ($state -eq "RUNNING") {
  gcloud compute ssh $Instance --project $Project --zone $Zone --quiet --command="tail -4 /opt/srwork/train.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
}
