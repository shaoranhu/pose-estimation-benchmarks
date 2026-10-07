param(
    [ValidateSet('prepare','smoke','full')][string]$Mode = 'full',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$cache = Join-Path $PSScriptRoot '.cache'
$env:PYTHONIOENCODING = 'utf-8'
if ($Mode -eq 'prepare') {
    & $Python (Join-Path $PSScriptRoot 'scripts/prepare_data.py') --cache $cache
} elseif ($Mode -eq 'smoke') {
    & $Python (Join-Path $PSScriptRoot 'scripts/evaluate.py') --cache $cache --output (Join-Path $PSScriptRoot 'results/smoke-rerun') --models yolo26n-pose --limit 8
} else {
    & $Python (Join-Path $PSScriptRoot 'scripts/evaluate.py') --cache $cache --output (Join-Path $PSScriptRoot 'results/full-rerun')
}
if ($LASTEXITCODE -ne 0) { throw "Pose experiment failed: exit code $LASTEXITCODE" }
