"""Speak faster without raising the pitch: streaming WSOLA time-scale modification.

TTS chunks arrive while the sentence is still being generated, so the stretcher is incremental:
`process(chunk)` returns whatever output is already final and `flush()` returns the rest. Feeding the
same audio in any chunking gives the same output as feeding it at once.
"""

import numpy as np


def adaptive_speed(lag_seconds: float, base: float, maximum: float, low: float = 1.5, high: float = 6.0) -> float:
    """Reading speed for the next sentence: `base` while on time, rising linearly to `maximum` as the
    listener falls behind (`lag_seconds` = how long the sentence has already waited + audio queued ahead)."""
    if maximum <= base or lag_seconds <= low:
        speed = base
    elif lag_seconds >= high:
        speed = maximum
    else:
        speed = base + (maximum - base) * (lag_seconds - low) / (high - low)
    return round(speed * 20) / 20   # steps of 0.05: no jitter between neighbouring sentences


class SpeechStretcher:
    def __init__(self, sample_rate: int, speed: float, frame_ms: float = 30.0, search_ms: float = 10.0):
        if speed < 1.0:
            raise ValueError("Only speeding up (speed >= 1) is supported")
        self.speed = float(speed)
        self.frame = int(sample_rate * frame_ms / 1000) // 2 * 2
        self.half = self.frame // 2                       # synthesis hop (50 % overlap)
        self.tolerance = int(sample_rate * search_ms / 1000)
        self.analysis_hop = self.half * self.speed
        n = np.arange(self.frame)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * n / self.frame)).astype(np.float32)   # periodic Hann, sums to 1
        self.first_window = self.window.copy()
        self.first_window[:self.half] = 1.0               # no fade-in on the very first samples
        self.buffer = np.zeros(0, np.float32)
        self.buffer_start = 0                              # absolute index of buffer[0]
        self.frame_index = 0
        self.previous = None                               # absolute start of the previously chosen frame
        self.overlap = np.zeros(self.frame, np.float32)

    # ---- public ----
    def process(self, chunk) -> np.ndarray:
        self.buffer = np.concatenate([self.buffer, np.asarray(chunk, np.float32)])
        return self._run(final=False)

    def flush(self) -> np.ndarray:
        out = self._run(final=True)
        return np.concatenate([out, self.overlap[:self.half]])

    # ---- internals ----
    def _segment(self, start: int, length: int) -> np.ndarray:
        """Input samples [start, start+length); zeros beyond the end of the input."""
        offset = start - self.buffer_start
        piece = self.buffer[max(offset, 0):max(offset + length, 0)]
        if len(piece) < length:
            piece = np.concatenate([piece, np.zeros(length - len(piece), np.float32)])
        return piece

    def _run(self, final: bool) -> np.ndarray:
        outputs = []
        total = self.buffer_start + len(self.buffer)
        while True:
            nominal = int(round(self.frame_index * self.analysis_hop))
            low = max(0, nominal - self.tolerance)
            high = nominal + self.tolerance
            if final:
                if nominal >= total:
                    break
            elif total < high + self.frame:
                break
            if self.previous is None:
                position = 0
            else:
                target = self._segment(self.previous + self.half, self.frame)
                region = self._segment(low, self.frame + high - low)
                correlation = np.correlate(region, target, "valid")
                squares = np.concatenate([[0.0], np.cumsum(region.astype(np.float64) ** 2)])
                energy = squares[self.frame:self.frame + len(correlation)] - squares[:len(correlation)]
                position = low + int(np.argmax(correlation / np.sqrt(energy + 1e-9)))
            window = self.first_window if self.frame_index == 0 else self.window
            self.overlap += self._segment(position, self.frame) * window
            outputs.append(self.overlap[:self.half].copy())
            self.overlap = np.concatenate([self.overlap[self.half:], np.zeros(self.half, np.float32)])
            self.previous = position
            self.frame_index += 1
            # Keep only what the next frame can still look at.
            keep_from = max(0, min(position + self.half, int(round(self.frame_index * self.analysis_hop)) - self.tolerance))
            drop = keep_from - self.buffer_start
            if drop > 0:
                self.buffer = self.buffer[drop:]
                self.buffer_start = keep_from
        return np.concatenate(outputs) if outputs else np.zeros(0, np.float32)
