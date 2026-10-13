# Runs translate-video with the project-local venv and its pip-installed CUDA 13 runtime.
# Overrides CUDA_PATH/PATH for this process only; system CUDA 12.8 is left untouched.
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)
$root = Split-Path -Parent $PSScriptRoot
$site = Join-Path $root ".venv\Lib\site-packages"
$cu13 = Join-Path $site "nvidia\cu13"
$cu13bin = Join-Path $cu13 "bin"

# The pip CUDA 13 wheel keeps DLLs in bin\x86_64, but CUDA_PATH consumers look in bin\.
foreach ($dll in Get-ChildItem (Join-Path $cu13bin "x86_64\*.dll")) {
    $target = Join-Path $cu13bin $dll.Name
    if (-not (Test-Path $target)) { Copy-Item $dll.FullName $target }
}

$env:CUDA_PATH = $cu13
$env:CUDA_HOME = $cu13
$clean = ($env:PATH -split ";" | Where-Object { $_ -and $_ -notmatch "NVIDIA GPU Computing Toolkit\\CUDA" }) -join ";"
$env:PATH = "$cu13bin;$cu13bin\x86_64;$site\torch\lib;$clean"
$env:VLLM_USE_FLASHINFER_SAMPLER = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TMPDIR = Join-Path $root ".cache\tmp"
$env:TRITON_CACHE_DIR = Join-Path $root ".cache\triton"
$env:VLLM_CACHE_ROOT = Join-Path $root ".cache\vllm"
New-Item -ItemType Directory -Force $env:TMPDIR, $env:TRITON_CACHE_DIR, $env:VLLM_CACHE_ROOT | Out-Null
& (Join-Path $root ".venv\Scripts\translate-video.exe") @Rest
exit $LASTEXITCODE
