"""Drive the real window + engine without a human: Start, play an English clip, Stop, close.

A window appears on the desktop and the clip is audible for ~30 s.
  .venv\Scripts\python scripts\app_selftest.py [--seconds 30]
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--clip", default=str(ROOT / "examples" / "english_10s.wav"))
    parser.add_argument("--clip-volume", type=float, default=0.25)
    args = parser.parse_args()

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from translate_video.app.client import EngineClient
    from translate_video.app.window import MainWindow

    app = QApplication([])
    app.setStyle("Fusion")
    window = MainWindow(EngineClient())
    window.show()
    started = time.monotonic()
    log, state = [], {"player": None, "phase": "start", "deadline": None}
    original = window.handle

    def traced(event):
        if event["kind"] != "level":
            log.append((time.monotonic() - started, event["kind"], window.state))
            summary = {k: v for k, v in event.items() if k not in ("kind", "event")}
            if event["kind"] == "pipeline":
                summary = {"type": event["event"]["type"], **{k: event["event"].get(k) for k in ("source", "translation", "text")}}
            print(f"{time.monotonic() - started:7.1f}s {event['kind']:12} state={window.state:10} {summary}", flush=True)
        original(event)

    window.handle = traced

    def tick():
        now = time.monotonic()
        if state["phase"] == "start":
            state["phase"] = "loading"
            window.toggle()
        elif state["phase"] == "loading" and window.state == "listening":
            state["phase"] = "listening"
            state["deadline"] = now + args.seconds
            state["player"] = subprocess.Popen([
                sys.executable, "-c",
                "import time,sounddevice as sd,soundfile as sf,sys;time.sleep(1);d,r=sf.read(sys.argv[1],dtype='float32');"
                "sd.play(d*float(sys.argv[2]),r);sd.wait()", args.clip, str(args.clip_volume)])
        elif state["phase"] == "listening" and now > state["deadline"]:
            state["phase"] = "stopping"
            window.toggle()
        elif state["phase"] == "stopping" and window.state == "ready":
            state["phase"] = "done"
            window.close()
            app.quit()
        elif window.state == "error":
            print("ERROR:", window.status.text())
            window.close()
            app.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(200)
    app.exec()
    if state["player"]:
        state["player"].terminate()
    print("final phase:", state["phase"], "| output dir:", window.output_dir)


if __name__ == "__main__":
    main()
