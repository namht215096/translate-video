# One-shot, project-local setup for Windows 10/11 + NVIDIA GPU (tested: RTX 5060 Ti, Python 3.12).
# Everything is installed into .venv\ and .cache\ in this folder; model weights use your HF cache.
# vLLM has no official Windows build, so this uses the community SystemPanic/vllm-windows wheel.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$vllm = "0.29.0"
$wheel = "vllm-$vllm+cu132-cp312-cp312-win_amd64.whl"
New-Item -ItemType Directory -Force .cache\dl, .cache\pip-tmp | Out-Null
$env:TMPDIR = Join-Path $root ".cache\pip-tmp"
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path ".cache\dl\$wheel")) {
    Invoke-WebRequest "https://github.com/SystemPanic/vllm-windows/releases/download/v$vllm/$wheel" -OutFile ".cache\dl\$wheel"
}
# Run sequentially: two pip processes on one venv corrupt it.
& $py -m pip install -U pip
& $py -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
& $py -m pip install ".cache\dl\$wheel" --extra-index-url https://download.pytorch.org/whl/cu130
& $py -m pip freeze --exclude-editable | Select-String -Pattern "^(torch|torchaudio|torchvision|transformers|numpy|huggingface.hub|tokenizers|safetensors|triton-windows)==" |
    ForEach-Object { $_.Line } | Set-Content .cache\constraints.txt
# qwen-asr/vieneu pin older libraries than vLLM-for-Windows needs; skip their pins, keep runtime deps.
& $py -m pip install --no-deps qwen-asr==0.0.6 vieneu==3.8.3
& $py -m pip install -c .cache\constraints.txt accelerate librosa soundfile sox nagisa soynlp pytz qwen-omni-utils `
    kaldi-native-fbank onnxruntime sea-g2p soxr sounddevice pytest flask gradio PySide6 PyAudioWPatch
& $py -m pip install --no-deps -e .
Write-Host "Desktop app: scripts\build_exe.ps1 builds TranslateVideo.exe (or run start_app.bat)."
Write-Host "Done. CLI: powershell -File scripts\run_windows.ps1 --input examples\english_10s.wav --realtime --save-audio --output output\my-run"
