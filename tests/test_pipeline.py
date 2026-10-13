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


def test_each_sentence_flows_translate_synthesize_play_in_order(tmp_path, monkeypatch):
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
    # stages overlap across sentences, but each sentence keeps its order and sentences never swap
    first = [c for c in calls if c in (("translate", "Hello,"), ("synthesize", "Xin chào,"), ("play", 0.1))]
    second = [c for c in calls if c in (("translate", "world."), ("synthesize", "thế giới."), ("play", 0.2))]
    assert first == [("translate", "Hello,"), ("synthesize", "Xin chào,"), ("play", 0.1)]
    assert second == [("translate", "world."), ("synthesize", "thế giới."), ("play", 0.2)]
    assert calls.index(("play", 0.1)) < calls.index(("play", 0.2))
    assert not list(output.glob("*.wav"))


def test_translation_and_tts_do_not_wait_for_the_speaker(tmp_path):
    import threading

    release = threading.Event()
    translated = []

    class CountingTranslator(Translator):
        def translate(self, text):
            result = super().translate(text)
            translated.append(text)
            if len(translated) == 2:
                release.set()        # both sentences are translated while the speaker is still busy
            return result

    class BlockingPlayer:
        name = "fake"

        def play(self, audio, sample_rate):
            assert release.wait(5), "translation of sentence 2 waited for the speaker"

    output = tmp_path / "run"
    Pipeline(ASR(), CountingTranslator(), TTS(), output, player=BlockingPlayer()).run(loud(1))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in events].count("translated") == 2
    assert [e["type"] for e in events].count("segment") == 2


def test_default_speed_shortens_audio_and_is_logged(tmp_path):
    class ToneTTS:
        sample_rate = 48000

        def synthesize(self, text):
            return (0.3 * np.sin(2 * np.pi * 180 * np.arange(48000) / 48000)).astype(np.float32)

    output = tmp_path / "run"
    Pipeline(ASR(), Translator(), ToneTTS(), output, speed=1.1, max_speed=1.4).run(loud(1))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    segments = [e for e in events if e["type"] == "segment"]
    assert segments and all(e["speed"] == 1.1 for e in segments)
    assert all(abs(e["audio_seconds"] - 1 / 1.1) < 0.05 for e in segments)


def test_speed_rises_with_backlog_up_to_the_maximum(tmp_path):
    import time

    class ToneTTS:
        sample_rate = 48000

        def synthesize(self, text):
            return (0.3 * np.sin(2 * np.pi * 180 * np.arange(48000) / 48000)).astype(np.float32)

    class SlowPlayer:
        name = "fake"

        def play(self, audio, sample_rate):
            time.sleep(0.6)

    class MoreWords(ASR):
        def feed(self, audio):
            return " ".join(f"w{i}," for i in range(8)), True

        def finish(self):
            return " ".join(f"w{i}," for i in range(8))

    class Echo(Translator):
        def translate(self, text):
            return text

    output = tmp_path / "run"
    Pipeline(MoreWords(), Echo(), ToneTTS(), output, player=SlowPlayer(), speed=1.1, max_speed=1.4,
             speed_low_lag=0.3, speed_high_lag=1.5).run(loud(2))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    speeds = [e["speed"] for e in events if e["type"] == "segment"]
    assert len(speeds) == 8
    assert speeds[0] == 1.1 and max(speeds) == 1.4 and speeds[-1] == 1.4
    assert all(1.1 <= v <= 1.4 for v in speeds)
    assert not [e for e in events if e["type"] == "dropped"]      # speeding up, never dropping


def test_stop_discards_the_queued_speech_quickly(tmp_path):
    import threading
    import time

    stop = threading.Event()

    class SlowPlayer:
        name = "fake"

        def play(self, audio, sample_rate):
            stop.set()
            time.sleep(0.3)

    def chunks():
        for item in loud(1):
            yield item
        stop.wait(2)
        yield loud(1)[0]

    started = time.monotonic()
    Pipeline(ASR(), Translator(), TTS(), tmp_path / "run", player=SlowPlayer(), stop_event=stop).run(chunks())
    assert time.monotonic() - started < 3


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


def test_streaming_audio_reaches_player_before_generation_finishes(tmp_path, monkeypatch):
    import threading

    first_played = threading.Event()
    played = []

    class StreamTTS:
        sample_rate = 48000

        def synthesize_stream(self, text):
            yield np.ones(480, dtype=np.float32) * .1
            assert first_played.wait(5), "first chunk was not played while the second was being generated"
            yield np.ones(480, dtype=np.float32) * .1

    class Player:
        name = "fake"

        def play_stream(self, chunks, sample_rate):
            assert sample_rate == 48000
            for chunk in chunks:
                played.append(len(chunk))
                first_played.set()

    monkeypatch.setattr("translate_video.pipeline.AudioPlayer", Player)
    output = tmp_path / "stream"
    Pipeline(ASR(), Translator(), StreamTTS(), output, play=True).run([
        (np.ones(1600, dtype=np.float32), .1),
    ])
    assert played == [480, 480] * 2
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    segments = [event for event in events if event["type"] == "segment"]
    assert all(event["audio_seconds"] == .02 for event in segments)
    assert all(event["tts_first_chunk_ms"] <= event["tts_stream_elapsed_ms"] for event in segments)
    assert all("tts_ms" not in event for event in segments)
    assert all(event["speed"] == 1.0 for event in segments)
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


def loud(n=1):
    return [(np.ones(1600, dtype=np.float32) * 0.1, 0.1 * (i + 1)) for i in range(n)]


def test_injected_player_and_on_event(tmp_path):
    played, seen = [], []

    class Player:
        name = "fake"

        def play(self, audio, sample_rate):
            played.append(len(audio))

    output = tmp_path / "run"
    Pipeline(ASR(), Translator(), TTS(), output, player=Player(), on_event=seen.append).run(loud(2))
    logged = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert played == [1600, 1600]
    assert [e["type"] for e in seen] == [e["type"] for e in logged]


def test_external_stop_ends_the_run_cleanly(tmp_path):
    import threading

    stop = threading.Event()
    asr = ASR()
    fed = []
    original = asr.feed
    asr.feed = lambda audio: (fed.append(1), original(audio))[1]

    def chunks():
        yield loud(1)[0]
        stop.set()
        yield loud(1)[0]
        yield loud(1)[0]

    Pipeline(asr, Translator(), TTS(), tmp_path / "run", stop_event=stop).run(chunks())
    assert len(fed) == 1


class RevisingASR(ASR):
    """Commits 'Hello, there.' and then contradicts it."""

    def feed(self, audio):
        self.calls += 1
        return ("Hello, there." if self.calls <= 2 else "Goodbye, now."), True

    def finish(self):
        return "Goodbye, now."


class Greeter(Translator):
    def translate(self, text):
        return text


def test_revision_stops_by_default_and_resets_for_live_use(tmp_path):
    with pytest.raises(RuntimeError, match="revised"):
        Pipeline(RevisingASR(), Greeter(), TTS(), tmp_path / "stop").run(loud(3))
    asr = RevisingASR()
    output = tmp_path / "reset"
    Pipeline(asr, Greeter(), TTS(), output, on_revision="reset").run(loud(3))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(e["type"] == "revision_reset" for e in events)
    assert events[-1]["type"] == "summary"
    assert [e["source"] for e in events if e["type"] == "segment"][0] == "Hello,"


def test_stale_segments_are_dropped_when_playback_lags(tmp_path):
    import time

    class SlowTranslator(Translator):
        def translate(self, text):
            time.sleep(0.4)
            return super().translate(text)

    output = tmp_path / "run"
    Pipeline(ASR(), SlowTranslator(), TTS(), output, max_lag_seconds=0.1).run(loud(2))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    kinds = sorted(e["type"] for e in events if e["type"] in ("segment", "dropped"))
    assert kinds == ["dropped", "segment"]


def test_segments_that_are_already_vietnamese_are_skipped(tmp_path):
    class VietnameseASR(ASR):
        def feed(self, audio):
            return "Chắc chắn là anh sẽ tìm kiếm.", True

        def finish(self):
            return "Chắc chắn là anh sẽ tìm kiếm."

    from translate_video.language import looks_vietnamese

    output = tmp_path / "run"
    Pipeline(VietnameseASR(), Greeter(), TTS(), output, skip_source=looks_vietnamese).run(loud(2))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not [e for e in events if e["type"] == "segment"]
    assert [e["type"] for e in events if e["type"] == "skipped"]


def test_a_failed_translation_ends_the_run_unless_errors_are_tolerated(tmp_path):
    class Flaky(Translator):
        def translate(self, text):
            if text == "Hello,":
                raise RuntimeError("Translation reached its token limit without completing")
            return super().translate(text)

    with pytest.raises(RuntimeError, match="token limit"):
        Pipeline(ASR(), Flaky(), TTS(), tmp_path / "strict").run(loud(2))
    output = tmp_path / "tolerant"
    Pipeline(ASR(), Flaky(), TTS(), output, tolerate_errors=True).run(loud(2))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in events].count("translation_failed") == 1
    assert [e["source"] for e in events if e["type"] == "segment"] == ["world."]
    assert events[-1]["type"] == "summary"


def test_a_failed_tts_sentence_is_skipped_when_tolerated(tmp_path):
    class FlakyTTS(TTS):
        def synthesize(self, text):
            if text == "Xin chào,":
                raise RuntimeError("no audio")
            return super().synthesize(text)

    output = tmp_path / "tolerant"
    Pipeline(ASR(), Translator(), FlakyTTS(), output, tolerate_errors=True).run(loud(2))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in events].count("tts_failed") == 1
    assert events[-1]["type"] == "summary"
    with pytest.raises(RuntimeError, match="no audio"):
        Pipeline(ASR(), Translator(), FlakyTTS(), tmp_path / "strict").run(loud(2))


def test_chinese_windows_use_a_longer_silence_limit(tmp_path):
    pipeline = Pipeline(ASR(), Translator(), TTS(), tmp_path / "run", silence_seconds=0.6, silence_seconds_cjk=1.0)
    assert pipeline.silence_limit("so we begin") == 0.6
    assert pipeline.silence_limit("所以今天这期视频我们不。") == 1.0
    assert Pipeline(ASR(), Translator(), TTS(), tmp_path / "x", silence_seconds=0.6).silence_limit("你好") == 0.6


def test_a_mid_sentence_pause_does_not_close_a_chinese_window(tmp_path):
    class ChineseASR(ASR):
        def feed(self, audio):
            return "所以今天这期视频我们不", True

    asr = ChineseASR()
    quiet = (np.zeros(1600, dtype=np.float32), 0.0)
    chunks = [(np.ones(1600, dtype=np.float32) * 0.1, 0.1)] + [(quiet[0], 0.1 * (i + 2)) for i in range(7)]
    Pipeline(asr, Translator(), TTS(), tmp_path / "short", silence_seconds=0.6, silence_seconds_cjk=1.0,
             on_revision="reset").run(chunks)
    resets_with_cjk_limit = asr.resets
    asr2 = ChineseASR()
    Pipeline(asr2, Translator(), TTS(), tmp_path / "plain", silence_seconds=0.6, on_revision="reset").run(chunks)
    assert resets_with_cjk_limit < asr2.resets          # 0.7 s of silence closes the window only with the 0.6 s limit
