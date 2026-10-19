"""Capture everything the PC plays (WASAPI loopback) while playing our own speech on the same device.

One persistent output stream plays silence when idle: it keeps loopback delivering frames and gives
`EchoCanceller` a continuous reference clock. Windows 10 before build 20438 has no per-process
loopback, so our own speech is removed by subtracting the known waveform instead.
"""

from collections import deque
import queue
import threading
import time

import numpy as np

from .echo import EchoCanceller

ASR_RATE = 16000


class PlaybackQueue:
    """Mono audio waiting to be rendered by the output callback (already scaled by the volume)."""

    def __init__(self):
        self.items = deque()
        self.lock = threading.Lock()
        self.drained = threading.Condition(self.lock)

    def push(self, audio):
        with self.lock:
            self.items.append(np.asarray(audio, np.float32))

    def render(self, frames: int):
        out = np.zeros(frames, np.float32)
        filled = 0
        with self.lock:
            while filled < frames and self.items:
                head = self.items[0]
                take = min(len(head), frames - filled)
                out[filled:filled + take] = head[:take]
                filled += take
                if take == len(head):
                    self.items.popleft()
                else:
                    self.items[0] = head[take:]
            if not self.items:
                self.drained.notify_all()
        return out

    def pending_frames(self) -> int:
        with self.lock:
            return sum(len(item) for item in self.items)

    def clear(self):
        with self.lock:
            self.items.clear()
            self.drained.notify_all()

    def wait_drained(self, stopped: threading.Event, poll=0.02):
        with self.lock:
            while self.items and not stopped.is_set():
                self.drained.wait(poll)


class SystemAudio:
    """Context manager exposing `chunks()` (cleaned 16 kHz mono) and `player` for `Pipeline`."""

    LOOKAHEAD_SECONDS = 0.4   # capture is cleaned this long after it arrives

    def __init__(self, gain=1.0, history_seconds=6.0):
        import pyaudiowpatch as pyaudio

        self.pyaudio = pyaudio
        self.gain = gain
        self.stopped = threading.Event()
        self.queue = PlaybackQueue()
        self.raw = queue.Queue(maxsize=1000)        # untouched loopback blocks from the audio thread
        self.captured = queue.Queue(maxsize=3000)   # cleaned 10 ms blocks (30 s of backlog at most)
        self.resync = False
        self.dropped_blocks = 0
        self.audio = pyaudio.PyAudio()
        loopback = self.audio.get_default_wasapi_loopback()
        wanted = loopback["name"].removesuffix(" [Loopback]")
        wasapi = self.audio.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        output = next(
            (self.audio.get_device_info_by_index(i) for i in range(self.audio.get_device_count())
             if self.audio.get_device_info_by_index(i)["hostApi"] == wasapi
             and self.audio.get_device_info_by_index(i)["maxOutputChannels"] > 0
             and self.audio.get_device_info_by_index(i)["name"] == wanted),
            None,
        )
        if output is None:
            raise RuntimeError(f"No output device matches loopback '{loopback['name']}'")
        self.device_name = output["name"]
        self.rate = int(output["defaultSampleRate"])
        self.out_channels = max(1, int(output["maxOutputChannels"]))
        self.in_channels = max(1, int(loopback["maxInputChannels"]))
        self.canceller = EchoCanceller(self.rate, history_seconds)
        self.gate_unlocked = True   # mute ASR input while our speech plays and the echo lag is unknown
        self.output_index, self.loopback_index = output["index"], loopback["index"]
        self.out_stream = self.in_stream = None
        self.workers = []
        self.glitches = {"capture": 0, "playback": 0}
        self.player = _Player(self)

    # -- lifecycle --
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self):
        pa = self.pyaudio
        self.in_stream = self.audio.open(
            format=pa.paFloat32, channels=self.in_channels, rate=self.rate, input=True,
            input_device_index=self.loopback_index, frames_per_buffer=self.rate // 100,
            stream_callback=self._capture_callback,
        )
        self.out_stream = self.audio.open(
            format=pa.paFloat32, channels=self.out_channels, rate=self.rate, output=True,
            output_device_index=self.output_index, frames_per_buffer=self.rate // 100,
            stream_callback=self._playback_callback,
        )
        self.in_stream.start_stream()
        self.out_stream.start_stream()
        self.workers = [threading.Thread(target=self._process_loop, daemon=True),
                        threading.Thread(target=self._monitor_lag, daemon=True)]
        for worker in self.workers:
            worker.start()

    def close(self):
        self.stopped.set()
        self.queue.clear()
        for stream in (self.out_stream, self.in_stream):
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        for worker in self.workers:
            worker.join(timeout=2)
        self.audio.terminate()

    # -- audio callbacks --
    def _playback_callback(self, in_data, frames, time_info, status):
        if status:
            self.glitches["playback"] += 1
        mono = np.clip(self.queue.render(frames), -1.0, 1.0)
        self.canceller.add_reference(mono)
        block = np.zeros((frames, self.out_channels), np.float32)
        block[:, :min(2, self.out_channels)] = mono[:, None]
        return block.tobytes(), self.pyaudio.paContinue

    def _capture_callback(self, in_data, frames, time_info, status):
        # Runs on the audio thread and competes for the GIL with model code: do nothing but hand over bytes.
        if status:
            self.glitches["capture"] += 1
            self.resync = True      # frames may have been lost: the reference clock can no longer be trusted
        try:
            self.raw.put_nowait(in_data)
        except queue.Full:
            self.glitches["capture"] += 1
            self.resync = True
        return None, self.pyaudio.paContinue

    def _process_loop(self):
        canceller = self.canceller
        delay = int(self.LOOKAHEAD_SECONDS * self.rate)
        since_track = 0
        while not self.stopped.is_set():
            try:
                data = self.raw.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.resync:
                self.resync = False
                canceller.lag = None
            block = np.frombuffer(data, np.float32).reshape(-1, self.in_channels)
            # Use the front L/R pair only: that is where the reference is played.
            start = canceller.ingest(block[:, :min(2, self.in_channels)].mean(axis=1))
            since_track += len(block)
            if since_track >= self.rate // 10:
                since_track = 0
                # The playback->capture delay can move by one audio period under load; follow it on
                # fresh audio while the blocks it applies to are still waiting out the lookahead.
                if canceller.locked and canceller.reference_energy(self.rate // 2) > 1e-3:
                    canceller.verify(tolerance_seconds=0.04, window_seconds=self.LOOKAHEAD_SECONDS)
            if start < delay:
                continue
            cleaned = canceller.clean(start - delay, len(block))
            if (self.gate_unlocked and not canceller.locked
                    and canceller.reference_energy(self.rate // 2) > 1e-4):
                cleaned = np.zeros_like(cleaned)
            try:
                self.captured.put_nowait(cleaned)
            except queue.Full:
                self.captured.get_nowait()   # keep the newest audio; the pipeline is far behind
                self.captured.put_nowait(cleaned)
                self.dropped_blocks += 1

    # -- echo lag management --
    def calibrate(self, attempts=3):
        """Play a short quiet noise burst and measure the playback->capture lag. Returns lag in ms or None."""
        rng = np.random.default_rng()   # fresh noise: a repeated burst would correlate with the old ones
        for _ in range(attempts):
            self.canceller.lag = None
            self.queue.push(rng.standard_normal(int(0.3 * self.rate)).astype(np.float32) * 0.08)
            self.queue.wait_drained(self.stopped)
            time.sleep(0.4)
            deadline = time.monotonic() + 3.0   # the capture must be processed up to now before measuring
            while self.raw.qsize() > 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.05)
            if self.canceller.calibrate(window_seconds=1.2, min_lag_seconds=-0.5, max_lag_seconds=1.0):
                return self.canceller.lag / self.rate * 1000
        return None

    def _monitor_lag(self):
        """While unlocked and speech is playing, look for the lag over the full range."""
        while not self.stopped.wait(0.5):
            if not self.canceller.locked and self.canceller.reference_energy(self.rate) > 1e-3:
                self.canceller.calibrate(window_seconds=1.5, min_lag_seconds=-0.5, max_lag_seconds=1.0)

    # -- consumer side --
    def chunks(self, block_seconds=0.1, idle_timeout=5.0):
        """Yield (16 kHz mono block, elapsed seconds) of everything playing except our own speech."""
        import soxr

        resampler = soxr.ResampleStream(self.rate, ASR_RATE, 1, dtype="float32")
        block = round(block_seconds * ASR_RATE)
        buffered = np.zeros(0, np.float32)
        end = 0.0
        last = time.monotonic()
        while not self.stopped.is_set():
            try:
                pieces = [self.captured.get(timeout=0.25)]
            except queue.Empty:
                if time.monotonic() - last > idle_timeout:
                    raise RuntimeError("Loopback stopped delivering audio")
                continue
            last = time.monotonic()
            while True:   # take whatever else is waiting so one wake-up handles a whole backlog
                try:
                    pieces.append(self.captured.get_nowait())
                except queue.Empty:
                    break
            buffered = np.concatenate([buffered, resampler.resample_chunk(np.concatenate(pieces))])
            while len(buffered) >= block:
                end += block / ASR_RATE
                yield buffered[:block].copy(), end
                buffered = buffered[block:]


class _Player:
    """Same interface as `AudioPlayer`; every segment goes through the persistent output stream."""

    name = "wasapi"

    def __init__(self, system: SystemAudio):
        self.system = system

    def _prepare(self, chunk, sample_rate):
        import soxr

        chunk = np.asarray(chunk, np.float32)
        if chunk.ndim != 1 or not np.isfinite(chunk).all():
            raise ValueError("Playback requires a finite mono waveform")
        if sample_rate != self.system.rate:
            chunk = soxr.resample(chunk, sample_rate, self.system.rate).astype(np.float32)
        return np.clip(chunk * self.system.gain, -1.0, 1.0)

    def play(self, audio, sample_rate):
        self.system.queue.push(self._prepare(audio, sample_rate))
        self.system.queue.wait_drained(self.system.stopped)

    def play_stream(self, chunks, sample_rate):
        try:
            for chunk in chunks:
                if self.system.stopped.is_set():
                    break
                self.system.queue.push(self._prepare(chunk, sample_rate))
            self.system.queue.wait_drained(self.system.stopped)
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()
