"""`python -m translate_video.app` / `translate-video-app`."""

import multiprocessing
import sys


def main():
    multiprocessing.freeze_support()
    from PySide6.QtWidgets import QApplication

    from .client import EngineClient
    from .window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(EngineClient())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
