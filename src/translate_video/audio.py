"""Audio input and direct playback of in-memory mono waveforms."""

import math
import queue
from pathlib import Path
import shutil
import subprocess
import time


class AudioPlayer:
    """Play an in-memory waveform directly, without writing an audio file."""

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

    def play(self, audio, sample_rate):
        import numpy as np

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Playback requires a finite mono waveform")
        if sample_rate <= 0:
            raise ValueError("Playback sample rate must be positive")
        audio = np.clip(audio, -1, 1)
        if self.command:
            if self.name == "pw-play":
                args = [self.command, "--raw", "--format", "s16", "--rate",
                        str(sample_rate), "--channels", "1", "-"]
            else:
                args = [self.command, "-q", "-t", "raw", "-f", "S16_LE",
                        "-r", str(sample_rate), "-c", "1", "-"]
            pcm = (audio * 32767).astype("<i2").tobytes()
            result = subprocess.run(
                args, input=pcm, capture_output=True,
            )
            if result.returncode:
                detail = result.stderr.decode(errors="replace").strip()
                raise RuntimeError(f"{self.name} playback failed: {detail}")
        else:
            self.sounddevice.play(audio, sample_rate, blocking=True)

    def play_stream(self, chunks, sample_rate):
        """Feed one player for an entire utterance; no application audio queue."""
        import numpy as np

        def pcm(chunk):
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim != 1 or not np.isfinite(chunk).all():
                raise ValueError("Playback requires a finite mono waveform")
            return (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()

        if sample_rate <= 0:
            raise ValueError("Playback sample rate must be positive")
        if not self.command:
            try:
                with self.sounddevice.RawOutputStream(
                    samplerate=sample_rate, channels=1, dtype="int16",
                ) as stream:
                    for chunk in chunks:
                        stream.write(pcm(chunk))
            finally:
                close = getattr(chunks, "close", None)
                if close:
                    close()
            return
        args = ([self.command, "--raw", "--format", "s16", "--rate",
                 str(sample_rate), "--channels", "1", "-"] if self.name == "pw-play"
                else [self.command, "-q", "-t", "raw", "-f", "S16_LE",
                      "-r", str(sample_rate), "-c", "1", "-"])
        # stderr uses the inherited terminal: no unconsumed pipe can deadlock.
        process = subprocess.Popen(args, stdin=subprocess.PIPE)
        try:
            for chunk in chunks:
                process.stdin.write(pcm(chunk))
                process.stdin.flush()
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"{self.name} streaming playback failed")
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        finally:
            if not process.stdin.closed:
                process.stdin.close()
            close = getattr(chunks, "close", None)
            if close:
                close()


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
