$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$candidates = @(
    (Join-Path $root "..\.venv\Scripts\python.exe"),
    (Join-Path $root ".venv\Scripts\python.exe")
)
$py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) { throw "Nie znaleziono python.exe w .venv" }

$common = @(
    "--max-epochs", "20",
    "--batch-size", "16",
    "--accelerator", "gpu",
    "--devices", "1"
)

Write-Host "=== #9 CNN classification bin=5 ===" -ForegroundColor Cyan
& $py (Join-Path $PSScriptRoot "train_macros.py") --head cnn --task classification --bin-size 5 @common
if ($LASTEXITCODE -ne 0) { throw "Job #9 failed" }

Write-Host "=== #10 CNN classification bin=10 ===" -ForegroundColor Cyan
& $py (Join-Path $PSScriptRoot "train_macros.py") --head cnn --task classification --bin-size 10 @common
if ($LASTEXITCODE -ne 0) { throw "Job #10 failed" }

Write-Host "DONE #9 i #10" -ForegroundColor Green
