"""Streaming ASR with one worker for translation, TTS, and direct playback."""

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time

from .segmentation import Segmenter
from .audio import AudioPlayer


@dataclass
class Segment:
    index: int
    text: str
    source_end_seconds: float
    committed_at: float


class Pipeline:
    def __init__(self, asr, translator, tts, output: Path, *, play=False,
                 silence_seconds=0.6, max_window_seconds=12.0,
                 rms_threshold=0.008, final_only=False, metadata=None):
        self.asr, self.translator, self.tts = asr, translator, tts
        self.output = output
        self.play = play
        self.player = AudioPlayer() if play else None
        self.silence_seconds = silence_seconds
        self.max_window_seconds = max_window_seconds
        self.rms_threshold = rms_threshold
        self.final_only = final_only
        self.metadata = metadata or {}
        self.text_queue = queue.Queue(maxsize=32)
        self.errors = queue.Queue()
        self.stopped = threading.Event()
        self.log_lock = threading.Lock()
        self.index = 0

    def log(self, event):
        with self.log_lock:
            self.log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.log_file.flush()

    def check(self):
        if not self.errors.empty():
            raise self.errors.get()

    def put(self, destination, item):
        while not self.stopped.is_set():
            self.check()
            try:
                destination.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        self.check()
        raise RuntimeError("Pipeline stopped")

    def emit(self, texts, end):
        for text in texts:
            self.index += 1
            segment = Segment(self.index, text, end, time.monotonic())
            # ASR must not block on a full downstream queue during capture.
            try:
                self.text_queue.put_nowait(segment)
            except queue.Full:
                raise RuntimeError("Translation backlog exceeds 32 segments; stopping") from None

    def translate_worker(self):
        try:
            if self.player:
                print(f"Playback backend: {self.player.name} (in-memory PCM)", flush=True)
            while not self.stopped.is_set():
                try:
                    segment = self.text_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if segment is None:
                    return
                started = time.monotonic()
                translation = self.translator.translate(segment.text)
                translated = time.monotonic()
                streaming = hasattr(self.tts, "synthesize_stream")
                stream_times = {}
                if streaming:
                    samples = 0

                    def measured_chunks():
                        nonlocal samples
                        import numpy as np

                        chunks = self.tts.synthesize_stream(translation)
                        try:
                            for chunk in chunks:
                                chunk = np.asarray(chunk, dtype=np.float32)
                                if chunk.ndim != 1 or not np.isfinite(chunk).all():
                                    raise ValueError("TTS returned an invalid mono waveform")
                                if not len(chunk):
                                    continue
                                now = time.monotonic()
                                stream_times.setdefault("first", now)
                                stream_times["last"] = now
                                samples += len(chunk)
                                yield chunk
                        finally:
                            close = getattr(chunks, "close", None)
                            if close:
                                close()

                    playback_started = time.monotonic()
                    if self.player:
                        self.player.play_stream(measured_chunks(), self.tts.sample_rate)
                    else:
                        for _ in measured_chunks():
                            pass
                    playback_finished = time.monotonic()
                    if not samples:
                        raise RuntimeError("TTS returned no audio")
                    ready = stream_times["last"]
                    audio_seconds = samples / self.tts.sample_rate
                else:
                    audio = self.tts.synthesize(translation)
                    ready = time.monotonic()
                    audio_seconds = len(audio) / self.tts.sample_rate
                event = {
                    "type": "segment", "index": segment.index,
                    "source": segment.text, "translation": translation,
                    "source_observed_end_seconds": segment.source_end_seconds,
                    "commit_elapsed_seconds": segment.committed_at - self.started_at,
                    "audio_ready_elapsed_seconds": ready - self.started_at,
                    "translation_queue_ms": (started - segment.committed_at) * 1000,
                    "translation_ms": (translated - started) * 1000,
                    "audio_seconds": audio_seconds,
                    "commit_to_audio_ready_ms": (ready - segment.committed_at) * 1000,
                }
                if streaming:
                    event.update({
                        "tts_first_chunk_ms": (stream_times["first"] - translated) * 1000,
                        "first_audio_ready_elapsed_seconds": stream_times["first"] - self.started_at,
                        "tts_stream_elapsed_ms": (ready - translated) * 1000,
                    })
                    if self.player:
                        event["playback_submit_elapsed_seconds"] = playback_started - self.started_at
                        event["playback_ms"] = (playback_finished - playback_started) * 1000
                else:
                    event["tts_ms"] = (ready - translated) * 1000
                print(f"[{event['index']}] {event['source']}\n  -> {event['translation']}", flush=True)
                if self.player and not streaming:
                    playback_started = time.monotonic()
                    event["playback_submit_elapsed_seconds"] = playback_started - self.started_at
                    self.player.play(audio, self.tts.sample_rate)
                    event["playback_ms"] = (time.monotonic() - playback_started) * 1000
                self.log(event)
        except BaseException as exc:
            self.errors.put(exc)
            self.stopped.set()

    def run(self, chunks):
        import numpy as np

        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "configuration.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        self.started_at = started
        with (self.output / "events.jsonl").open("w", encoding="utf-8") as self.log_file:
            worker = threading.Thread(target=self.translate_worker)
            worker.start()
            segmenter = Segmenter()
            window = silence = 0.0
            active = False
            end = 0.0

            def finish_window():
                nonlocal segmenter, window, silence, active
                if active:
                    text = self.asr.finish()
                    self.emit(segmenter.update(text, final=True), end)
                self.asr.reset()
                segmenter = Segmenter()
                window = silence = 0.0
                active = False

            interrupted = False
            try:
                try:
                    for audio, end in chunks:
                        self.check()
                        duration = len(audio) / 16000
                        voiced = float(np.sqrt(np.mean(audio ** 2))) >= self.rms_threshold
                        if not active and not voiced:
                            continue
                        active = True
                        window += duration
                        silence = 0.0 if voiced else silence + duration
                        tick = time.monotonic()
                        text, changed = self.asr.feed(audio)
                        if changed:
                            self.log({"type": "partial", "text": text,
                                      "source_observed_end_seconds": end,
                                      "asr_call_ms": (time.monotonic() - tick) * 1000})
                            if not self.final_only:
                                self.emit(segmenter.update(text), end)
                        if silence >= self.silence_seconds or window >= self.max_window_seconds:
                            finish_window()
                except KeyboardInterrupt:
                    interrupted = True
                finish_window()
                self.put(self.text_queue, None)
                worker.join()
                self.check()
                self.log({"type": "summary", "segments": self.index,
                          "source_seconds": end, "wall_seconds": time.monotonic() - started,
                          "interrupted": interrupted})
            finally:
                self.stopped.set()
                worker.join()
