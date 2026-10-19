import numpy as np

from translate_video.echo import EchoCanceller

RATE = 16000
BLOCK = RATE // 10


def scenario(seconds=8, lag=2400, seed=1):
    rng = np.random.default_rng(seed)
    n = seconds * RATE
    video = (rng.standard_normal(n) * 0.05).astype(np.float32)
    ref = np.zeros(n, np.float32)
    envelope = (np.sin(np.arange(n) / RATE * 2 * np.pi * 0.7) > 0).astype(np.float32)  # on/off speech-like bursts
    ref[:] = rng.standard_normal(n).astype(np.float32) * 0.2 * envelope
    echo = np.zeros(n, np.float32)
    echo[lag:] = ref[:n - lag]
    return video, ref, video + echo


def feed(canceller, ref, capture, start=0, stop=None):
    out = []
    for b in range(start, (stop or len(ref)) // BLOCK):
        canceller.add_reference(ref[b * BLOCK:(b + 1) * BLOCK])
        out.append(canceller.process(capture[b * BLOCK:(b + 1) * BLOCK]))
    return np.concatenate(out)


def test_calibrates_and_removes_echo():
    video, ref, capture = scenario()
    canceller = EchoCanceller(RATE)
    feed(canceller, ref, capture, stop=3 * RATE)
    assert canceller.calibrate()
    assert canceller.lag == 2400
    cleaned = feed(canceller, ref, capture, start=30)
    expected = video[3 * RATE:]
    assert np.sqrt(np.mean((cleaned - expected) ** 2)) < 1e-5
    assert np.sqrt(np.mean(capture[3 * RATE:] ** 2)) > 3 * np.sqrt(np.mean((cleaned - expected) ** 2))


def test_unlocked_passes_audio_through():
    video, ref, capture = scenario()
    canceller = EchoCanceller(RATE)
    assert not canceller.locked
    cleaned = feed(canceller, ref, capture, stop=RATE)
    assert np.array_equal(cleaned, capture[:RATE])


def test_silent_reference_does_not_calibrate():
    video, _, _ = scenario()
    canceller = EchoCanceller(RATE)
    feed(canceller, np.zeros_like(video), video, stop=3 * RATE)
    assert not canceller.calibrate()
    assert canceller.lag is None


def test_verify_tracks_a_lag_change():
    video, ref, capture = scenario(seconds=12)
    canceller = EchoCanceller(RATE)
    feed(canceller, ref, capture, stop=3 * RATE)
    assert canceller.calibrate() and canceller.lag == 2400
    # A capture dropout shifts the counters: the same echo now arrives 800 frames later.
    _, _, shifted = scenario(seconds=12, lag=3200)
    feed(canceller, ref, shifted, start=30, stop=7 * RATE)
    assert canceller.verify()
    assert canceller.lag == 3200


def test_rejects_uncorrelated_capture():
    rng = np.random.default_rng(3)
    ref = (rng.standard_normal(4 * RATE) * 0.2).astype(np.float32)
    capture = (rng.standard_normal(4 * RATE) * 0.2).astype(np.float32)
    canceller = EchoCanceller(RATE)
    feed(canceller, ref, capture)
    assert not canceller.calibrate()


def test_follows_a_lag_jump_with_lookahead():
    """The playback->capture delay moves by one audio period at t=4 s; cleaning lags capture by 0.4 s."""
    seconds, jump = 10, 4 * RATE
    rng = np.random.default_rng(5)
    n = seconds * RATE
    video = (rng.standard_normal(n) * 0.05).astype(np.float32)
    envelope = (np.sin(np.arange(n) / RATE * 2 * np.pi * 0.7) > -0.3).astype(np.float32)
    ref = (rng.standard_normal(n) * 0.2 * envelope).astype(np.float32)
    echo = np.zeros(n, np.float32)
    for t in range(n):
        lag = 2400 if t < jump else 2880
        if t >= lag:
            echo[t] = ref[t - lag]
    capture = video + echo
    canceller = EchoCanceller(RATE)
    delay = int(0.4 * RATE)
    cleaned = np.zeros(n, np.float32)
    for b in range(n // BLOCK):
        canceller.add_reference(ref[b * BLOCK:(b + 1) * BLOCK])
        start = canceller.ingest(capture[b * BLOCK:(b + 1) * BLOCK])
        if canceller.locked and canceller.reference_energy(RATE // 2) > 1e-3:
            canceller.verify(tolerance_seconds=0.04, window_seconds=0.4)
        elif not canceller.locked and b == 30:
            assert canceller.calibrate()
        if canceller.locked and start >= delay:
            cleaned[start - delay:start - delay + BLOCK] = canceller.clean(start - delay, BLOCK)
    settled = slice(jump + RATE // 2, n - delay)
    residual = np.sqrt(np.mean((cleaned[settled] - video[settled]) ** 2))
    assert residual < 0.005    # ~ -40 dB relative to the 0.2 echo
    assert canceller.lag == 2880
