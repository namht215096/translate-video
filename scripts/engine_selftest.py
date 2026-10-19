"""Headless end-to-end check of the app engine on real devices (what the GUI drives).

Loads the models, starts listening, plays an English clip from another process (standing in for a
YouTube tab), and prints what the engine heard/translated. Audible on the default output for ~30 s.

  .venv\\Scripts\\python scripts\\engine_selftest.py [examples\\english_10s.wav] [--gain 1.0] [--seconds 40]
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default=str(ROOT / "examples" / "english_10s.wav"))
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--clip-volume", type=float, default=0.2)
    parser.add_argument("--seconds", type=float, default=40)
    args = parser.parse_args()

    from translate_video.app.engine import Settings, engine_main
    from dataclasses import asdict

    context = mp.get_context("spawn")
    commands, events = context.Queue(), context.Queue()
    process = context.Process(target=engine_main, args=(commands, events, str(ROOT)), daemon=True)
    process.start()
    settings = asdict(Settings(gain=args.gain))
    commands.put(("load", settings))
    started = time.monotonic()
    ready = False
    player = None
    log = []
    deadline = None
    while True:
        try:
            event = events.get(timeout=0.5)
        except Exception:
            event = None
        now = time.monotonic()
        if event:
            kind = event["kind"]
            if kind == "level":
                continue
            log.append((now - started, event))
            print(f"{now - started:7.1f}s", {k: v for k, v in event.items() if k != "kind"} | {"kind": kind}, flush=True)
            if kind == "ready":
                ready = True
                commands.put(("start", settings))
            if kind == "error":
                break
            if kind == "status" and event.get("state") == "listening":
                time.sleep(1.0)
                player = subprocess.Popen([
                    sys.executable, "-c",
                    "import sounddevice as sd,soundfile as sf,sys;d,r=sf.read(sys.argv[1],dtype='float32');"
                    "sd.play(d*float(sys.argv[2]),r);sd.wait()", args.clip, str(args.clip_volume)])
                deadline = now + args.seconds
            if kind == "status" and event.get("state") == "idle" and deadline:
                break
        if deadline and now > deadline:
            commands.put(("stop",))
            deadline = now + 120
        if not process.is_alive():
            print("engine died")
            break
    if player:
        player.terminate()
    commands.put(("shutdown",))
    process.join(timeout=20)
    segments = [e["event"] for _, e in log if e["kind"] == "pipeline" and e["event"]["type"] == "segment"]
    print(f"\n{len(segments)} translated segments; engine log: {ROOT / '.cache' / 'engine.log'}")
    for s in segments:
        print(" ", s["source"], "->", s["translation"])


if __name__ == "__main__":
    main()
