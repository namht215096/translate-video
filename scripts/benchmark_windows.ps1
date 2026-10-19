# Runs the full pipeline on the sample audio with both TTS backends and writes timing logs.
# Usage: powershell -File scripts\benchmark_windows.ps1 [-Audio examples\english_10s.wav] [-NoRealtime]
param([string] $Audio = "examples\english_10s.wav", [switch] $NoRealtime)
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$base = Join-Path $root "output\bench-$stamp"
New-Item -ItemType Directory -Force $base | Out-Null
$runs = @()
foreach ($backend in "vieneu", "mms") {
    $dir = Join-Path $base $backend
    $extra = @()
    if (-not $NoRealtime) { $extra += "--realtime" }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & (Join-Path $PSScriptRoot "run_windows.ps1") --input (Join-Path $root $Audio) --tts-backend $backend `
        --save-audio --output $dir @extra 2>&1 | Out-File -Encoding utf8 (Join-Path $base "$backend.console.log")
    $watch.Stop()
    $line = "$backend process_total_seconds=$([math]::Round($watch.Elapsed.TotalSeconds, 2)) (includes startup + model loading)"
    $line | Out-File -Encoding utf8 -Append (Join-Path $base "process-times.txt")
    Write-Host $line
    $runs += $dir
}
$env:PYTHONUTF8 = "1"
& (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "compare_runs.py") @runs |
    Out-File -Encoding utf8 (Join-Path $base "comparison.md")
Get-Content (Join-Path $base "comparison.md")
Write-Host "Logs and audio: $base"
