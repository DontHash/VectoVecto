# build_release.ps1 — build the Pro desktop bundle (one-folder, zipped).
#
# Not part of the v1.1.0 OSS release (CLI + web app ship instead). Commercial
# builds must exclude the CC-BY-NC-SA photo weights — see docs/RELEASE.md.
#
#   powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not $SkipInstall) {
        python -m pip install --upgrade pyinstaller
    }
    python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])" |
        ForEach-Object { $version = $_.Trim() }
    Write-Host "building vectovecto $version"
    pyinstaller packaging/vectovecto.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

    $zip = "dist/vectovecto-$version-win64.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path "dist/vectovecto" -DestinationPath $zip
    Write-Host "wrote $zip"
    Write-Host "smoke test: dist/vectovecto/vectovecto.exe --help"
} finally {
    Pop-Location
}
