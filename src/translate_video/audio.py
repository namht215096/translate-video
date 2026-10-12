"""File replay and bounded microphone capture at 16 kHz mono."""

import math
import queue
from pathlib import Path
import shutil
import subprocess
import time


class WavePlayer:
    """Play completed segments through the desktop audio server when available."""

    def __init__(self):
        self.command = shutil.which("pw-play") or shutil.which("aplay")
        self.sounddevice = None
        if self.command:
            self.name = Path(self.command).name
        else:
            try:
                import sounddevice as sd
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Playback requires pw-play, aplay, or the PortAudio runtime"
                ) from exc
            self.sounddevice = sd
            self.name = "sounddevice"

    def play(self, path):
        if self.command:
            result = subprocess.run(
                [self.command, str(path)], capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(f"{self.name} playback failed: {result.stderr.strip()}")
        else:
            import soundfile as sf
            audio, rate = sf.read(path, dtype="float32")
            self.sounddevice.play(audio, rate, blocking=True)


def file_chunks(path, block_seconds=0.1, realtime=False):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    block = round(block_seconds * 16000)
    started = time.monotonic()
    for start in range(0, len(audio), block):
        chunk = audio[start:start + block]
        end = (start + len(chunk)) / 16000
        if realtime:
            time.sleep(max(0, started + end - time.monotonic()))
        yield chunk, end


def microphone_chunks(device=None, block_seconds=0.1):
    import sounddevice as sd

    pending = queue.Queue(maxsize=50)
    failures = queue.Queue()

    def callback(indata, frames, timing, status):
        if status:
            failures.put(RuntimeError(f"Microphone capture failed: {status}"))
        try:
            pending.put_nowait(indata[:, 0].copy())
        except queue.Full:
            failures.put(RuntimeError("Microphone backlog exceeded 5 seconds; audio was lost"))

    end = 0.0
    with sd.InputStream(
        samplerate=16000, channels=1, dtype="float32", device=device,
        blocksize=round(block_seconds * 16000), callback=callback,
    ):
        while True:
            if not failures.empty():
                raise failures.get()
            try:
                chunk = pending.get(timeout=0.25)
            except queue.Empty:
                continue
            end += len(chunk) / 16000
            yield chunk, end
