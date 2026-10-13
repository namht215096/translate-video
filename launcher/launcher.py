"""TranslateVideo.exe: finds the project-local .venv next to the exe and starts the app without a console."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path


def message(text, flags=0x10):
    ctypes.windll.user32.MessageBoxW(0, text, "Translate Video", flags)


def main():
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    python = root / ".venv" / "Scripts" / "pythonw.exe"
    if not python.exists():
        message(f"Không tìm thấy môi trường Python:\n{python}\n\nChạy scripts\setup_windows.ps1 trước.")
        return 1
    if not (root / "src" / "translate_video" / "app" / "__main__.py").exists():
        message(f"Thiếu mã nguồn app trong:\n{root}\src")
        return 1
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: the app outlives this launcher.
    subprocess.Popen([str(python), "-m", "translate_video.app"], cwd=root,
                     creationflags=0x00000008 | 0x00000200, close_fds=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
