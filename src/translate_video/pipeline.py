"""Streaming ASR feeding three independent stages: translation, TTS, and playback.

    ASR (main thread) -> text_queue -> translate -> tts_queue -> TTS -> play_queue -> playback

Every stage starts the next piece of work as soon as it is free, so a slow speaker never delays
translation or synthesis. Instead the reading speed rises with the delay the listener has accumulated
(see `stretch.adaptive_speed`); no sentence is dropped unless `max_lag_seconds` is set explicitly.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import threading
import time

from .audio import AudioPlayer
from .segmentation import RevisionError, Segmenter
from .stretch import SpeechStretcher, adaptive_speed


@dataclass
class Segment:
    index: int
    text: str
    source_end_seconds: float
    committed_at: float


@dataclass
class Job:
    """One sentence travelling through translation, TTS and playback."""
    segment: Segment
    translate_started: float = 0.0
    translated_at: float = 0.0
    translation: str = ""
    tts_started: float = 0.0
    first_chunk_at: float | None = None
    ready_at: float = 0.0
    speed: float = 1.0
    lag_seconds: float = 0.0
    audio_seconds: float = 0.0
    submit_at: float | None = None
    finished_at: float = 0.0
    chunks: queue.Queue = field(default_factory=queue.Queue)
    saved: list = field(default_factory=list)


class _Stopped(Exception):
    """Raised inside a worker when the pipeline is shutting down; not an error."""


class Pipeline:
    def __init__(self, asr, translator, tts, output: Path, *, play=False,
                 silence_seconds=0.6, max_window_seconds=12.0,
                 rms_threshold=0.008, final_only=False, metadata=None, save_audio=False,
                 player=None, on_event=None, stop_event=None, on_revision="stop",
                 max_lag_seconds=None, skip_source=None,
                 speed=1.0, max_speed=None, speed_low_lag=1.5, speed_high_lag=6.0,
                 segmentation="punctuation"):
        self.asr, self.translator, self.tts = asr, translator, tts
        self.output = output
        self.play = play
        self.save_audio = save_audio
        self.player = player if player is not None else (AudioPlayer() if play else None)
        self.on_event = on_event
        self.external_stop = stop_event
        self.on_revision = on_revision
        self.max_lag_seconds = max_lag_seconds
        self.skip_source = skip_source
        self.speed = speed                                   # base reading speed; may be changed while running
        self.max_speed = speed if max_speed is None else max_speed
        self.speed_low_lag = speed_low_lag
        self.speed_high_lag = speed_high_lag
        self.segmentation = segmentation
        self.silence_seconds = silence_seconds
        self.max_window_seconds = max_window_seconds
        self.rms_threshold = rms_threshold
        self.final_only = final_only
        self.metadata = metadata or {}
        self.text_queue = queue.Queue(maxsize=32)
        self.tts_queue = queue.Queue()
        self.play_queue = queue.Queue()
        self.errors = queue.Queue()
        self.stopped = threading.Event()
        self.log_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.audio_enqueued = 0.0        # seconds of speech handed to playback so far
        self.audio_finished = 0.0        # seconds of speech whose playback has completed
        self.playing = None              # (job, monotonic start) while a sentence is being played
        self.index = 0

    # ---- shared helpers ----
    def log(self, event):
        with self.log_lock:
            self.log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.log_file.flush()
        if self.on_event:
            self.on_event(event)

    def write_audio(self, index, chunks):
        import numpy as np
        import soundfile

        folder = self.output / "audio"
        folder.mkdir(exist_ok=True)
        path = folder / f"segment_{index:03d}.wav"
        soundfile.write(path, np.concatenate(chunks), self.tts.sample_rate)
        return path.name

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

    def take(self, source):
        """Next item from a stage queue; raises _Stopped once the pipeline is shutting down."""
        while True:
            try:
                return source.get(timeout=0.1)
            except queue.Empty:
                if self.stopped.is_set():
                    raise _Stopped from None

    def new_segmenter(self):
        return Segmenter(mode=self.segmentation)

    def emit(self, texts, end):
        for text in texts:
            self.index += 1
            segment = Segment(self.index, text, end, time.monotonic())
            # ASR must not block on a full downstream queue during capture.
            try:
                self.text_queue.put_nowait(segment)
            except queue.Full:
                raise RuntimeError("Translation backlog exceeds 32 segments; stopping") from None

    def playback_backlog(self) -> float:
        """Seconds of already synthesized speech the listener still has to sit through."""
        with self.audio_lock:
            remaining = self.audio_enqueued - self.audio_finished
            if self.playing:
                job, started = self.playing
                remaining -= min(time.monotonic() - started, job.audio_seconds)
        return max(0.0, remaining)

    def choose_speed(self, lag_seconds):
        return adaptive_speed(lag_seconds, self.speed, self.max_speed, self.speed_low_lag, self.speed_high_lag)

    def fail(self, exc):
        self.errors.put(exc)
        self.stopped.set()

    # ---- stage 1: translation ----
    def translate_worker(self):
        try:
            while True:
                segment = self.take(self.text_queue)
                if segment is None:
                    self.tts_queue.put(None)
                    return
                if self.skip_source and self.skip_source(segment.text):
                    # e.g. already Vietnamese: our own speech heard again, or the wrong source language
                    self.log({"type": "skipped", "index": segment.index, "source": segment.text})
                    continue
                started = time.monotonic()
                if self.max_lag_seconds is not None and started - segment.committed_at > self.max_lag_seconds:
                    self.log({"type": "dropped", "index": segment.index, "source": segment.text,
                              "waited_ms": (started - segment.committed_at) * 1000})
                    continue
                job = Job(segment, translate_started=started)
                job.translation = self.translator.translate(segment.text)
                job.translated_at = time.monotonic()
                print(f"[{segment.index}] {segment.text}\n  -> {job.translation}", flush=True)
                self.log({"type": "translated", "index": segment.index, "source": segment.text,
                          "translation": job.translation,
                          "translation_queue_ms": (started - segment.committed_at) * 1000,
                          "translation_ms": (job.translated_at - started) * 1000})
                self.tts_queue.put(job)      # straight on to TTS; this thread moves on to the next sentence
        except _Stopped:
            pass
        except BaseException as exc:
            self.fail(exc)

    # ---- stage 2: speech synthesis (with adaptive speed) ----
    def tts_worker(self):
        try:
            while True:
                job = self.take(self.tts_queue)
                if job is None:
                    self.play_queue.put(None)
                    return
                self.synthesize(job)
        except _Stopped:
            pass
        except BaseException as exc:
            self.fail(exc)

    def synthesize(self, job):
        import numpy as np

        rate = self.tts.sample_rate
        job.tts_started = time.monotonic()
        # Delay the listener has accumulated: time since the sentence ended + speech still queued ahead of it.
        job.lag_seconds = (job.tts_started - job.segment.committed_at) + self.playback_backlog()
        job.speed = self.choose_speed(job.lag_seconds)
        stretcher = SpeechStretcher(rate, job.speed) if job.speed > 1.0 else None
        self.play_queue.put(job)             # playback waits for the first chunk of this job

        def deliver(chunk):
            if len(chunk) == 0:
                return
            if job.first_chunk_at is None:
                job.first_chunk_at = time.monotonic()
            seconds = len(chunk) / rate
            with self.audio_lock:
                job.audio_seconds += seconds
                self.audio_enqueued += seconds
            if self.save_audio:
                job.saved.append(chunk)
            job.chunks.put(chunk)

        try:
            if hasattr(self.tts, "synthesize_stream"):
                chunks = self.tts.synthesize_stream(job.translation)
                try:
                    for chunk in chunks:
                        chunk = np.asarray(chunk, dtype=np.float32)
                        if chunk.ndim != 1 or not np.isfinite(chunk).all():
                            raise ValueError("TTS returned an invalid mono waveform")
                        deliver(stretcher.process(chunk) if stretcher else chunk)
                finally:
                    close = getattr(chunks, "close", None)
                    if close:
                        close()
            else:
                audio = np.asarray(self.tts.synthesize(job.translation), dtype=np.float32)
                deliver(stretcher.process(audio) if stretcher else audio)
            if stretcher:
                deliver(stretcher.flush())
            if job.first_chunk_at is None:
                raise RuntimeError("TTS returned no audio")
        finally:
            job.ready_at = time.monotonic()
            job.chunks.put(None)             # always release the playback stage

    # ---- stage 3: playback ----
    def play_worker(self):
        try:
            if self.player:
                print(f"Playback backend: {self.player.name} (in-memory PCM)", flush=True)
            while True:
                job = self.take(self.play_queue)
                if job is None:
                    return
                self.play_job(job)
        except _Stopped:
            pass
        except BaseException as exc:
            self.fail(exc)

    def mark_submitted(self, job):
        job.submit_at = time.monotonic()
        with self.audio_lock:
            self.playing = (job, job.submit_at)

    def job_chunks(self, job):
        """Chunks of one sentence as the TTS stage produces them; marks the moment audio reaches the speaker."""
        while True:
            chunk = self.take(job.chunks)
            if chunk is None:
                return
            if job.submit_at is None:
                self.mark_submitted(job)
            yield chunk

    def play_job(self, job):
        streaming = hasattr(self.tts, "synthesize_stream")
        rate = self.tts.sample_rate
        chunks = self.job_chunks(job)
        if self.player and streaming:
            self.player.play_stream(chunks, rate)
        elif self.player:
            import numpy as np

            audio = np.concatenate(list(chunks))     # whole-waveform players need the complete sentence
            self.mark_submitted(job)
            self.player.play(audio, rate)
        else:
            for _ in chunks:
                pass
        job.finished_at = time.monotonic()
        with self.audio_lock:
            self.audio_finished += job.audio_seconds
            self.playing = None
        self.check()
        self.log(self.segment_event(job, streaming))

    def segment_event(self, job, streaming):
        segment = job.segment
        started_at = self.started_at
        event = {
            "type": "segment", "index": segment.index,
            "source": segment.text, "translation": job.translation,
            "source_observed_end_seconds": segment.source_end_seconds,
            "commit_elapsed_seconds": segment.committed_at - started_at,
            "audio_ready_elapsed_seconds": job.ready_at - started_at,
            "translation_queue_ms": (job.translate_started - segment.committed_at) * 1000,
            "translation_ms": (job.translated_at - job.translate_started) * 1000,
            "tts_queue_ms": (job.tts_started - job.translated_at) * 1000,
            "audio_seconds": job.audio_seconds,
            "speed": job.speed,
            "lag_at_tts_ms": job.lag_seconds * 1000,
            "commit_to_audio_ready_ms": (job.ready_at - segment.committed_at) * 1000,
        }
        first = job.first_chunk_at if job.first_chunk_at is not None else job.ready_at
        if streaming:
            event.update({
                "tts_first_chunk_ms": (first - job.tts_started) * 1000,
                "first_audio_ready_elapsed_seconds": first - started_at,
                "tts_stream_elapsed_ms": (job.ready_at - job.tts_started) * 1000,
            })
        else:
            event["tts_ms"] = (job.ready_at - job.tts_started) * 1000
        if job.submit_at is not None:
            event["playback_submit_elapsed_seconds"] = job.submit_at - started_at
            event["commit_to_speaker_ms"] = (job.submit_at - segment.committed_at) * 1000
            event["playback_wait_ms"] = max(0.0, (job.submit_at - first) * 1000)
            if self.player:
                event["playback_ms"] = (job.finished_at - job.submit_at) * 1000
        if self.save_audio and job.saved:
            event["audio_file"] = self.write_audio(segment.index, job.saved)
        return event

    # ---- ASR side ----
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
            workers = [threading.Thread(target=target, name=target.__name__)
                       for target in (self.translate_worker, self.tts_worker, self.play_worker)]
            for worker in workers:
                worker.start()
            segmenter = self.new_segmenter()
            window = silence = 0.0
            active = False
            end = 0.0

            def commit(text, final=False):
                nonlocal segmenter, window, silence, active
                try:
                    self.emit(segmenter.update(text, final=final), end)
                except RevisionError:
                    if self.on_revision != "reset":
                        raise
                    # Live use: drop this window's text and resynchronize instead of ending the session.
                    self.log({"type": "revision_reset", "text": text})
                    self.asr.reset()
                    segmenter = self.new_segmenter()
                    window = silence = 0.0
                    active = False

            def finish_window():
                nonlocal segmenter, window, silence, active
                if active:
                    commit(self.asr.finish(), final=True)
                self.asr.reset()
                segmenter = self.new_segmenter()
                window = silence = 0.0
                active = False

            interrupted = False
            aborted = False
            try:
                try:
                    for audio, end in chunks:
                        self.check()
                        if self.external_stop is not None and self.external_stop.is_set():
                            aborted = True
                            break
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
                                commit(text)
                        if silence >= self.silence_seconds or window >= self.max_window_seconds:
                            finish_window()
                except KeyboardInterrupt:
                    interrupted = True
                if aborted:
                    # Stop pressed: discard queued speech instead of reading it all out first.
                    interrupted = True
                    self.stopped.set()
                else:
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
