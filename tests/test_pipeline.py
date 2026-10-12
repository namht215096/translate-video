import json
import wave

import numpy as np
import pytest

from translate_video.audio import file_chunks
from translate_video.pipeline import Pipeline


class ASR:
    def __init__(self):
        self.reset()
        self.resets = 0

    def reset(self):
        self.calls = 0
        self.resets = getattr(self, "resets", 0) + 1

    def feed(self, audio):
        self.calls += 1
        return "Hello, world.", True

    def finish(self):
        return "Hello, world."


class Translator:
    def translate(self, text):
        return {"Hello,": "Xin chào,", "world.": "thế giới."}[text]


class TTS:
    sample_rate = 16000

    def synthesize(self, text):
        return np.zeros(1600, dtype=np.float32)


def test_stream_drain_no_duplicate_and_audio_order(tmp_path):
    output = tmp_path / "run"
    asr = ASR()
    Pipeline(asr, Translator(), TTS(), output).run([
        (np.ones(1600, dtype=np.float32) * 0.1, 0.1),
        (np.ones(1600, dtype=np.float32) * 0.1, 0.2),
    ])
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    segments = [event for event in events if event["type"] == "segment"]
    assert [event["source"] for event in segments] == ["Hello,", "world."]
    assert events[-1]["segments"] == 2
    with wave.open(str(output / "translated.wav")) as audio:
        assert audio.getnframes() == 3200
        assert audio.getframerate() == 16000


def test_worker_failure_propagates(tmp_path):
    class BrokenTranslator:
        def translate(self, text):
            raise ValueError("translation failed")

    with pytest.raises(ValueError, match="translation failed"):
        Pipeline(ASR(), BrokenTranslator(), TTS(), tmp_path / "run").run([
            (np.ones(1600, dtype=np.float32), 0.1),
        ])


def test_silence_flush_and_window_reset(tmp_path):
    asr = ASR()
    Pipeline(asr, Translator(), TTS(), tmp_path / "run", silence_seconds=0.2).run([
        (np.ones(1600, dtype=np.float32), 0.1),
        (np.zeros(1600, dtype=np.float32), 0.2),
        (np.zeros(1600, dtype=np.float32), 0.3),
        (np.zeros(1600, dtype=np.float32), 0.4),
    ])
    assert asr.resets >= 2


def test_stereo_file_is_downmixed_resampled_and_tail_preserved(tmp_path):
    import soundfile as sf

    path = tmp_path / "sample.wav"
    sf.write(path, np.ones((12000, 2), dtype=np.float32) * 0.1, 48000)
    chunks = list(file_chunks(path))
    assert sum(len(chunk) for chunk, _ in chunks) == 4000
    assert chunks[-1][1] == 0.25
    assert all(chunk.ndim == 1 for chunk, _ in chunks)
