"""Environment for the community vLLM Windows build; mirrors scripts/run_windows.ps1.

Must run before torch/vLLM are imported. Affects only this process and its children.
"""

import os
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def configure(root: Path = PROJECT_ROOT):
    site = Path(sys.prefix) / "Lib" / "site-packages"
    cu13 = site / "nvidia" / "cu13"
    cu13_bin = cu13 / "bin"
    if cu13_bin.is_dir():
        # The pip CUDA 13 wheel keeps DLLs in bin\x86_64, but CUDA_PATH consumers look in bin\.
        for dll in (cu13_bin / "x86_64").glob("*.dll"):
            target = cu13_bin / dll.name
            if not target.exists():
                shutil.copy2(dll, target)
        os.environ["CUDA_PATH"] = os.environ["CUDA_HOME"] = str(cu13)
        keep = [p for p in os.environ.get("PATH", "").split(os.pathsep)
                if p and "NVIDIA GPU Computing Toolkit\\CUDA" not in p]
        dirs = [str(cu13_bin), str(cu13_bin / "x86_64"), str(site / "torch" / "lib")]
        os.environ["PATH"] = os.pathsep.join(dirs + keep)
        for directory in dirs:
            if os.path.isdir(directory):
                os.add_dll_directory(directory)
    cache = root / ".cache"
    for name, sub in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"), ("VLLM_CACHE_ROOT", "vllm")):
        os.environ[name] = str(cache / sub)
        (cache / sub).mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"   # FlashInfer's sampler JIT-compiles with nvcc + MSVC
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
