"""Streaming ASR producer with independent translation and audio workers."""

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
import wave

from .segmentation import Segmenter
from .audio import WavePlayer


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
        self.player = WavePlayer() if play else None
        self.silence_seconds = silence_seconds
        self.max_window_seconds = max_window_seconds
        self.rms_threshold = rms_threshold
        self.final_only = final_only
        self.metadata = metadata or {}
        self.text_queue = queue.Queue(maxsize=32)
        self.audio_queue = queue.Queue(maxsize=32)
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
            while not self.stopped.is_set():
                try:
                    segment = self.text_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if segment is None:
                    self.put(self.audio_queue, None)
                    return
                started = time.monotonic()
                translation = self.translator.translate(segment.text)
                translated = time.monotonic()
                audio = self.tts.synthesize(translation)
                ready = time.monotonic()
                event = {
                    "type": "segment", "index": segment.index,
                    "source": segment.text, "translation": translation,
                    "source_observed_end_seconds": segment.source_end_seconds,
                    "commit_elapsed_seconds": segment.committed_at - self.started_at,
                    "audio_ready_elapsed_seconds": ready - self.started_at,
                    "translation_queue_ms": (started - segment.committed_at) * 1000,
                    "translation_ms": (translated - started) * 1000,
                    "tts_ms": (ready - translated) * 1000,
                    "audio_seconds": len(audio) / self.tts.sample_rate,
                    "commit_to_audio_ready_ms": (ready - segment.committed_at) * 1000,
                }
                self.put(self.audio_queue, (audio, event, ready))
        except BaseException as exc:
            self.errors.put(exc)
            self.stopped.set()

    def audio_worker(self):
        import numpy as np
        import soundfile as sf

        try:
            if self.player:
                print(f"Playback backend: {self.player.name}", flush=True)
            with wave.open(str(self.output / "translated.wav"), "wb") as combined:
                combined.setnchannels(1)
                combined.setsampwidth(2)
                combined.setframerate(self.tts.sample_rate)
                while not self.stopped.is_set():
                    try:
                        item = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        return
                    audio, event, ready = item
                    segment_path = self.output / f"segment-{event['index']:04d}.wav"
                    sf.write(segment_path, audio, self.tts.sample_rate)
                    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
                    combined.writeframes(pcm.tobytes())
                    event["audio_queue_ms"] = (time.monotonic() - ready) * 1000
                    event["audio_sink_elapsed_seconds"] = time.monotonic() - self.started_at
                    self.log(event)
                    print(f"[{event['index']}] {event['source']}\n  -> {event['translation']}", flush=True)
                    if self.player:
                        self.player.play(segment_path)
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
            workers = [threading.Thread(target=self.translate_worker), threading.Thread(target=self.audio_worker)]
            for worker in workers:
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
                for worker in workers:
                    worker.join()
                self.check()
                self.log({"type": "summary", "segments": self.index,
                          "source_seconds": end, "wall_seconds": time.monotonic() - started,
                          "interrupted": interrupted})
            finally:
                self.stopped.set()
                for worker in workers:
                    worker.join()
