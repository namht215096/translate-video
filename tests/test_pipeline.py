import json

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


def test_stream_drain_no_duplicate_and_no_saved_audio(tmp_path):
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
    assert {path.name for path in output.iterdir()} == {"configuration.json", "events.jsonl"}
    assert all("audio_queue_ms" not in event for event in segments)


def test_playback_is_direct_and_finishes_before_next_translation(tmp_path, monkeypatch):
    calls = []

    class OrderedTranslator(Translator):
        def translate(self, text):
            calls.append(("translate", text))
            return super().translate(text)

    class OrderedTTS(TTS):
        def synthesize(self, text):
            calls.append(("synthesize", text))
            return np.full(1600, 0.1 if text == "Xin chào," else 0.2, dtype=np.float32)

    class Player:
        name = "fake"

        def play(self, audio, sample_rate):
            assert sample_rate == 16000
            assert isinstance(audio, np.ndarray)
            calls.append(("play", round(float(audio[0]), 1)))

    monkeypatch.setattr("translate_video.pipeline.AudioPlayer", Player)
    output = tmp_path / "run"
    Pipeline(ASR(), OrderedTranslator(), OrderedTTS(), output, play=True).run([
        (np.ones(1600, dtype=np.float32), 0.1),
    ])
    assert calls == [
        ("translate", "Hello,"), ("synthesize", "Xin chào,"), ("play", 0.1),
        ("translate", "world."), ("synthesize", "thế giới."), ("play", 0.2),
    ]
    assert not list(output.glob("*.wav"))


def test_playback_failure_propagates_without_hanging(tmp_path, monkeypatch):
    class Player:
        name = "fake"

        def play(self, audio, sample_rate):
            raise RuntimeError("speaker failed")

    monkeypatch.setattr("translate_video.pipeline.AudioPlayer", Player)
    with pytest.raises(RuntimeError, match="speaker failed"):
        Pipeline(ASR(), Translator(), TTS(), tmp_path / "run", play=True).run([
            (np.ones(1600, dtype=np.float32), 0.1),
        ])


def test_worker_failure_propagates(tmp_path):
    class BrokenTranslator:
        def translate(self, text):
            raise ValueError("translation failed")

    with pytest.raises(ValueError, match="translation failed"):
        Pipeline(ASR(), BrokenTranslator(), TTS(), tmp_path / "run").run([
            (np.ones(1600, dtype=np.float32), 0.1),
        ])


def test_streaming_audio_reaches_player_before_next_chunk(tmp_path, monkeypatch):
    calls = []

    class StreamTTS:
        sample_rate = 48000

        def synthesize_stream(self, text):
            for i in range(2):
                calls.append(("generate", i))
                yield np.ones(480, dtype=np.float32) * .1

    class Player:
        name = "fake"

        def play_stream(self, chunks, sample_rate):
            assert sample_rate == 48000
            for chunk in chunks:
                calls.append(("play", len(chunk)))

    monkeypatch.setattr("translate_video.pipeline.AudioPlayer", Player)
    output = tmp_path / "stream"
    Pipeline(ASR(), Translator(), StreamTTS(), output, play=True).run([
        (np.ones(1600, dtype=np.float32), .1),
    ])
    assert calls == [("generate", 0), ("play", 480),
                     ("generate", 1), ("play", 480)] * 2
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    segments = [event for event in events if event["type"] == "segment"]
    assert all(event["audio_seconds"] == .02 for event in segments)
    assert all(event["tts_first_chunk_ms"] <= event["tts_stream_elapsed_ms"] for event in segments)
    assert all("tts_ms" not in event for event in segments)
    assert {path.name for path in output.iterdir()} == {"configuration.json", "events.jsonl"}


def test_empty_tts_stream_fails_without_hanging(tmp_path):
    class EmptyTTS:
        sample_rate = 48000

        def synthesize_stream(self, text):
            return iter([])

    with pytest.raises(RuntimeError, match="no audio"):
        Pipeline(ASR(), Translator(), EmptyTTS(), tmp_path / "empty").run([
            (np.ones(1600, dtype=np.float32), .1),
        ])


def test_short_segments_are_emitted_before_end_of_unpunctuated_input(tmp_path):
    class LongASR(ASR):
        def feed(self, audio):
            return "one two three four five six seven eight nine", True

        def finish(self):
            return "one two three four five six seven eight nine"

    class IdentityTranslator:
        def translate(self, text):
            return text

    pipeline = Pipeline(LongASR(), IdentityTranslator(), TTS(), tmp_path / "short",
                        max_segment_words=8)

    def chunks():
        yield np.ones(1600, dtype=np.float32), .1
        yield np.ones(1600, dtype=np.float32), .2
        # The segment must be emitted before the generator ends/finalizes ASR.
        assert pipeline.index == 1

    pipeline.run(chunks())
    events = [json.loads(line) for line in (tmp_path / "short/events.jsonl").read_text().splitlines()]
    assert [event["source"] for event in events if event["type"] == "translation"] == [
        "one two three four five six seven eight", "nine",
    ]


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
