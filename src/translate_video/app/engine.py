"""Engine process: loads the models once, then runs capture -> ASR -> translation -> TTS sessions.

Commands arrive on a multiprocessing queue as tuples; results go back as dict events:
  ("load", settings) ("start", settings) ("stop",) ("gain", value) ("speed", value) ("reset_context",) ("context", pairs) ("shutdown",)
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
import threading
import time
import traceback


@dataclass
class Settings:
    gain: float = 1.0
    rms_threshold: float = 0.008
    silence_seconds: float = 0.6
    max_window_seconds: float = 12.0
    chunk_seconds: float = 1.0
    gpu_memory_utilization: float = 0.45
    speed: float = 1.2            # base reading speed
    max_speed: float = 1.8        # reached when the listener falls behind
    segmentation: str = "punctuation"
    speed_low_lag: float = 1.0    # delay (s) where speed starts to rise above `speed` ...
    speed_high_lag: float = 3.5   # ... and where it reaches `max_speed`
    source_language: str = "auto" # "auto" lets Qwen detect it; or a Qwen language name such as "English"/"Chinese"
    silence_seconds_cjk: float = 1.0
    save_audio: bool = False
    tts_voice: str | None = None
    model_device: str = "cuda:0"
    zh_translator: str = "opus"   # "opus" = Helsinki-NLP/opus-mt-zh-vi for Chinese sentences, "nllb" = NLLB for everything
    context_sentences: int = 3    # previous sentences given to the translator as context


def engine_main(commands, events, root: str):
    root = Path(root)
    (root / ".cache").mkdir(exist_ok=True)
    log = open(root / ".cache" / "engine.log", "a", encoding="utf-8", buffering=1)
    sys.setswitchinterval(0.001)   # audio callbacks share the GIL with model code; hand it over faster
    sys.stdout = sys.stderr = log   # pythonw has no console; keep vLLM/torch output for diagnosis
    from . import windows_env
    windows_env.configure(root)
    Engine(events, root).serve(commands)


class Engine:
    def __init__(self, events, root: Path):
        self.events = events
        self.root = root
        self.models = None
        self.session = None

    def send(self, kind, **data):
        self.events.put({"kind": kind, **data})

    def serve(self, commands):
        self.send("engine", state="started")
        while True:
            command = commands.get()
            name, args = command[0], command[1:]
            try:
                if name == "shutdown":
                    self.stop()
                    break
                getattr(self, name)(*args)
            except Exception as exc:  # keep serving; the UI shows the message
                traceback.print_exc()
                self.send("error", message=f"{type(exc).__name__}: {exc}")
        self.send("engine", state="stopped")

    def load(self, settings: dict):
        if self.models:
            self.send("ready", **self.models["info"])
            return
        settings = Settings(**settings)
        import torch
        from ..models import NLLBTranslator, OpusTranslator, QwenASR, VieNeuTTS

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("Cần GPU NVIDIA có CUDA")
        seconds = {}
        mark = time.monotonic()
        self.send("loading", stage="asr", step=1, steps=3)
        asr = QwenASR("Qwen/Qwen3-ASR-1.7B", None, settings.chunk_seconds, settings.gpu_memory_utilization)
        seconds["asr"] = time.monotonic() - mark
        mark = time.monotonic()
        self.send("loading", stage="nllb", step=2, steps=3)
        translator = NLLBTranslator("eng_Latn", "vie_Latn", settings.model_device)
        opus = OpusTranslator("Helsinki-NLP/opus-mt-zh-vi", settings.model_device)
        seconds["nllb"] = time.monotonic() - mark
        mark = time.monotonic()
        self.send("loading", stage="tts", step=3, steps=3)
        tts = VieNeuTTS("pnnbao-ump/VieNeu-TTS-v3-Turbo", settings.model_device, settings.tts_voice)
        seconds["tts"] = time.monotonic() - mark
        info = {"load_seconds": seconds, "gpu": torch.cuda.get_device_name(0)}
        self.models = {"asr": asr, "translator": translator, "opus": opus, "tts": tts, "info": info}
        self.send("ready", **info)

    def start(self, settings: dict):
        if self.session and self.session.thread.is_alive():
            return
        if not self.models:
            raise RuntimeError("Model chưa được tải")
        self.session = Session(self, Settings(**settings))
        self.session.thread.start()

    def stop(self):
        if self.session:
            self.session.stop()
            self.session.thread.join(timeout=30)
            self.session = None

    def gain(self, value: float):
        if self.session:
            self.session.set_gain(value)

    def speed(self, value: float):
        if self.session:
            self.session.set_speed(value)

    def reset_context(self):
        if self.session:
            self.session.reset_context()

    def context(self, pairs):
        if self.session:
            self.session.set_context(pairs)


class Session:
    """One listening session: owns the audio streams and a Pipeline run."""

    def __init__(self, engine: Engine, settings: Settings):
        self.engine = engine
        self.settings = settings
        self.stop_event = threading.Event()
        self.system = None
        self.pipeline = None
        self.thread = threading.Thread(target=self.run, name="session", daemon=True)

    def set_gain(self, value):
        if self.system:
            self.system.gain = value

    def set_speed(self, value):
        if self.pipeline:
            self.pipeline.speed = value
            self.pipeline.max_speed = max(self.pipeline.max_speed, value)

    def reset_context(self):
        if self.pipeline:
            self.pipeline.reset_context()

    def set_context(self, pairs):
        if self.pipeline:
            self.pipeline.set_context(pairs)

    def stop(self):
        self.stop_event.set()
        if self.system:
            self.system.stopped.set()

    def forward(self, event):
        self.engine.send("pipeline", event=event)

    def levels(self, chunks):
        import numpy as np

        for count, (audio, end) in enumerate(chunks):
            self.engine.send("level", rms=float(np.sqrt(np.mean(audio ** 2))))
            if count % 50 == 0 and self.pipeline and self.system:
                canceller = self.system.canceller
                self.pipeline.log({
                    "type": "audio_stats", "elapsed_seconds": end, "echo_locked": canceller.locked,
                    "echo_lag_ms": None if canceller.lag is None else canceller.lag / canceller.rate * 1000,
                    "glitches": dict(self.system.glitches), "dropped_blocks": self.system.dropped_blocks,
                })
            yield audio, end

    def run(self):
        from ..language import looks_vietnamese
        from ..loopback import SystemAudio
        from ..models import RoutingTranslator
        from ..pipeline import Pipeline

        send, models, settings = self.engine.send, self.engine.models, self.settings
        try:
            send("status", state="starting")
            with SystemAudio(gain=settings.gain) as system:
                self.system = system
                send("status", state="calibrating", device=system.device_name)
                lag = system.calibrate()
                send("calibration", lag_ms=lag, device=system.device_name)
                # Forcing "English" on a Chinese video makes Qwen invent English words; "auto" detects the language.
                models["asr"].set_language(None if settings.source_language == "auto" else settings.source_language)
                models["asr"].reset()
                output = self.engine.root / "output" / datetime.now().strftime("app-%Y%m%d-%H%M%S")
                translator = models["translator"]
                if settings.zh_translator == "opus":
                    translator = RoutingTranslator(models["opus"], translator)
                pipeline = self.pipeline = Pipeline(
                    models["asr"], translator, models["tts"], output,
                    player=system.player, on_event=self.forward, stop_event=self.stop_event,
                    on_revision="reset", skip_source=looks_vietnamese,
                    speed=settings.speed, max_speed=settings.max_speed, segmentation=settings.segmentation,
                    speed_low_lag=settings.speed_low_lag, speed_high_lag=settings.speed_high_lag,
                    silence_seconds_cjk=settings.silence_seconds_cjk, tolerate_errors=True, context_sentences=settings.context_sentences,
                    silence_seconds=settings.silence_seconds, max_window_seconds=settings.max_window_seconds,
                    rms_threshold=settings.rms_threshold, save_audio=settings.save_audio,
                    metadata={"arguments": {**asdict(settings), "app": True, "tts_backend": "vieneu", "realtime": True},
                              "model_load_seconds": models["info"]["load_seconds"], "gpu": models["info"]["gpu"],
                              "echo": {"device": system.device_name, "rate": system.rate, "lag_ms": lag}},
                )
                if self.stop_event.is_set():
                    return
                send("status", state="listening", output=str(output))
                pipeline.run(self.levels(system.chunks()))
        except Exception as exc:
            traceback.print_exc()
            send("error", message=f"{type(exc).__name__}: {exc}")
        finally:
            self.system = None
            send("status", state="idle")
