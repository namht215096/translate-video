"""Remove our own playback from a loopback capture.

Loopback is a digital copy of what the audio engine mixes, so the TTS we play comes back
as an exact (gain 1.0), constant-delay copy plus whatever else is playing. Subtracting the
reference at the right lag leaves only the other audio. The lag is a fixed offset between
two frame counters (frames handed to the output stream vs. frames captured), measured by
cross-correlation.
"""

import threading

import numpy as np


class EchoCanceller:
    def __init__(self, rate: int, history_seconds: float = 6.0):
        self.rate = rate
        self.size = int(rate * history_seconds)
        self.reference = np.zeros(self.size, np.float32)
        self.captured = np.zeros(self.size, np.float32)
        self.out_frames = 0   # frames of reference written so far
        self.cap_frames = 0   # frames of capture received so far
        self.lag = None       # capture frame c carries reference frame c - lag
        self.lock = threading.Lock()

    @property
    def locked(self) -> bool:
        return self.lag is not None

    def reset(self):
        with self.lock:
            self.reference[:] = 0
            self.captured[:] = 0
            self.out_frames = self.cap_frames = 0
            self.lag = None

    # -- ring helpers (call with the lock held) --
    def _write(self, ring, start, block):
        n = len(block)
        pos = start % self.size
        first = min(n, self.size - pos)
        ring[pos:pos + first] = block[:first]
        if first < n:
            ring[:n - first] = block[first:]

    def _read(self, ring, start, length, limit):
        """Frames [start, start+length); frames outside the history or not yet written are zero."""
        out = np.zeros(length, np.float32)
        lo = max(start, limit - self.size, 0)
        hi = min(start + length, limit)
        if hi > lo:
            idx = np.arange(lo, hi) % self.size
            out[lo - start:hi - start] = ring[idx]
        return out

    def add_reference(self, block):
        """Mono frames exactly as handed to the output stream."""
        block = np.asarray(block, np.float32)
        with self.lock:
            self._write(self.reference, self.out_frames, block)
            self.out_frames += len(block)

    def reference_energy(self, frames: int) -> float:
        with self.lock:
            return float(np.sqrt(np.mean(self._read(self.reference, self.out_frames - frames, frames, self.out_frames) ** 2)))

    def ingest(self, block) -> int:
        """Store a capture block; returns its starting frame number."""
        block = np.asarray(block, np.float32)
        with self.lock:
            start = self.cap_frames
            self._write(self.captured, start, block)
            self.cap_frames += len(block)
        return start

    def clean(self, start: int, length: int):
        """Capture frames [start, start+length) with our own playback removed (raw until the lag is known).

        Frames may be cleaned some time after they were ingested, so the lag can be tracked on later audio.
        """
        with self.lock:
            captured = self._read(self.captured, start, length, self.cap_frames)
            if self.lag is None:
                return captured
            ref = self._read(self.reference, start - self.lag, length, self.out_frames)
        return captured - ref

    def process(self, block):
        """Ingest and clean immediately (no lookahead)."""
        start = self.ingest(block)
        return self.clean(start, len(block))

    def estimate_lag(self, window_seconds=1.5, min_lag_seconds=-1.0, max_lag_seconds=2.0,
                     min_peak=0.02, min_ratio=7.0):
        """Best lag (frames) over [min, max], or None if the reference is silent or no clear peak stands out.

        Other audio playing at the same time lowers the absolute correlation, so a peak is accepted when
        it is `min_ratio` times the typical (rms) correlation over all lags rather than at a fixed level.
        """
        with self.lock:
            length = min(int(window_seconds * self.rate), self.cap_frames, self.size // 2)
            if length < self.rate // 4:
                return None
            start = self.cap_frames - length
            dmin, dmax = int(min_lag_seconds * self.rate), int(max_lag_seconds * self.rate)
            cap = self._read(self.captured, start, length, self.cap_frames)
            ref = self._read(self.reference, start - dmax, length + dmax - dmin, self.out_frames)
        if np.sum(ref ** 2) < 1e-6 or np.sum(cap ** 2) < 1e-9:
            return None
        n = 1 << int(np.ceil(np.log2(len(ref) + length)))
        corr = np.fft.irfft(np.fft.rfft(ref, n) * np.conj(np.fft.rfft(cap, n)), n)[:dmax - dmin + 1]
        cum = np.concatenate([[0.0], np.cumsum(ref.astype(np.float64) ** 2)])
        shifted = cum[length:length + dmax - dmin + 1] - cum[:dmax - dmin + 1]
        ncc = corr / np.sqrt(np.maximum(shifted, 1e-12) * np.sum(cap.astype(np.float64) ** 2))
        k = int(np.argmax(ncc))
        if ncc[k] < min_peak or ncc[k] < min_ratio * float(np.sqrt(np.mean(ncc ** 2))):
            return None
        return dmax - k

    def calibrate(self, **kwargs) -> bool:
        lag = self.estimate_lag(**kwargs)
        if lag is None:
            return False
        with self.lock:
            self.lag = lag
        return True

    def verify(self, tolerance_seconds=0.1, window_seconds=1.5) -> bool:
        """Re-measure near the current lag; adopt the new one if it moved. False if no evidence."""
        if self.lag is None:
            return self.calibrate(window_seconds=window_seconds)
        tol = tolerance_seconds
        current = self.lag / self.rate
        lag = self.estimate_lag(window_seconds=window_seconds, min_lag_seconds=current - tol,
                                max_lag_seconds=current + tol)
        if lag is None:
            return False
        with self.lock:
            self.lag = lag
        return True
