import time

import numpy as np
import pytest

from translate_video.stretch import SpeechStretcher, adaptive_speed

RATE = 48000


def tone(seconds=2.0, hz=200.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def stretch(audio, speed, chunk=None):
    stretcher = SpeechStretcher(RATE, speed)
    if chunk is None:
        pieces = [stretcher.process(audio)]
    else:
        pieces = [stretcher.process(audio[i:i + chunk]) for i in range(0, len(audio), chunk)]
    return np.concatenate(pieces + [stretcher.flush()])


@pytest.mark.parametrize("speed", [1.1, 1.25, 1.4])
def test_duration_shrinks_by_the_speed_and_pitch_is_unchanged(speed):
    audio = tone(2.0, 200.0)
    out = stretch(audio, speed)
    assert abs(len(out) / RATE - 2.0 / speed) < 0.05
    spectrum = np.abs(np.fft.rfft(out[RATE // 10:-RATE // 10] * np.hanning(len(out) - RATE // 5)))
    assert abs(np.argmax(spectrum) * RATE / (len(out) - RATE // 5) - 200.0) < 5.0


def test_output_is_continuous_and_not_louder():
    out = stretch(tone(2.0, 200.0), 1.4)
    assert np.max(np.abs(np.diff(out))) < 0.12        # a clean 200 Hz tone moves < 0.0105 per sample
    assert np.max(np.abs(out)) < 0.45
    assert abs(float(np.sqrt(np.mean(out[2000:-2000] ** 2))) - 0.4 / np.sqrt(2)) < 0.03


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(0)
    audio = (np.sin(2 * np.pi * 150 * np.arange(RATE) / RATE) * np.abs(np.sin(np.arange(RATE) / 4000))
             + rng.standard_normal(RATE) * 0.01).astype(np.float32)
    whole = stretch(audio, 1.3)
    for chunk in (480, 4800, 17000):
        assert np.array_equal(stretch(audio, 1.3, chunk), whole)


def test_streaming_emits_audio_before_the_input_ends():
    stretcher = SpeechStretcher(RATE, 1.2)
    first = stretcher.process(tone(0.5))
    assert len(first) > 0.3 * RATE / 1.2 - 2000


def test_speed_below_one_is_rejected_and_one_is_near_identity_length():
    with pytest.raises(ValueError):
        SpeechStretcher(RATE, 0.9)
    audio = tone(1.0)
    assert abs(len(stretch(audio, 1.0)) - len(audio)) < 1500


def test_empty_and_tiny_input():
    stretcher = SpeechStretcher(RATE, 1.3)
    assert len(stretcher.process(np.zeros(0, np.float32))) == 0
    assert len(stretcher.flush()) <= 1500
    assert len(stretch(np.ones(10, np.float32) * 0.1, 1.3)) < 1500


def test_real_time_factor_is_far_below_one():
    audio = tone(4.0)
    started = time.perf_counter()
    stretch(audio, 1.4, chunk=9600)
    assert time.perf_counter() - started < 1.0   # 4 s of audio


def test_adaptive_speed_curve():
    assert adaptive_speed(0.0, 1.1, 1.4) == 1.1
    assert adaptive_speed(1.0, 1.1, 1.4) == 1.1
    assert adaptive_speed(3.5, 1.1, 1.4) == 1.4
    assert adaptive_speed(10.0, 1.1, 1.4) == 1.4
    middle = adaptive_speed(2.25, 1.1, 1.4)
    assert 1.2 < middle < 1.3
    assert adaptive_speed(9.0, 1.1, 1.1) == 1.1                 # adaptation off
    values = [adaptive_speed(lag / 4, 1.1, 1.4) for lag in range(40)]
    assert values == sorted(values)
    assert all(abs(v * 20 - round(v * 20)) < 1e-9 for v in values)
