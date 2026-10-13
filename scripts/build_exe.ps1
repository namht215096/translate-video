# Builds TranslateVideo.exe (a ~10 MB launcher) in the project root using a throwaway build env.
# The exe only starts .venv\Scripts\pythonw.exe -m translate_video.app, so it works on this machine only.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env = Join-Path $root ".cache\buildenv"
if (-not (Test-Path $env)) { python -m venv $env }
& "$env\Scripts\python.exe" -m pip install -q pyinstaller
& "$env\Scripts\pyinstaller.exe" --noconfirm --onefile --noconsole --name TranslateVideo `
    --distpath $root --workpath (Join-Path $root ".cache\build") --specpath (Join-Path $root ".cache\build") `
    (Join-Path $root "launcher\launcher.py")
Write-Host "Built $root\TranslateVideo.exe"
