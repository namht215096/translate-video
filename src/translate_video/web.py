"""Local GPU web application. Audio exists only in browser/server memory."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
import queue
import struct
import threading
import time
from urllib.parse import urlsplit
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import Pipeline


class ModelRuntime:
    """Load once, keep one GPU session active, and reset ASR between files."""

    def __init__(self):
        self.loaded = False

    def prepare(self):
        if not self.loaded:
            import torch
            from .models import NLLBTranslator, QwenASR, VieNeuTTS

            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required to run the speech models.")
            torch.set_num_threads(2)
            self.asr = QwenASR("Qwen/Qwen3-ASR-1.7B", "English", .5, .55)
            self.translator = NLLBTranslator("eng_Latn", "vie_Latn", "cuda:0")
            self.tts = VieNeuTTS("pnnbao-ump/VieNeu-TTS-v3-Turbo", "cuda:0")
            # Initialize CUDA graphs and first-inference kernels before playback.
            self.asr.feed(np.zeros(8000, dtype=np.float32))
            self.asr.finish()
            self.translator.translate("Hello.")
            for _ in self.tts.synthesize_stream("Xin chào."):
                pass
            self.loaded = True
        self.asr.reset()

    def close(self):
        if self.loaded:
            self.tts.close()


class BrowserBridge:
    """Worker-thread to WebSocket bridge with browser playback acknowledgements."""

    name = "web-browser"

    def __init__(self, socket, loop, stopped):
        self.socket, self.loop, self.stopped = socket, loop, stopped
        self.pending = {}
        self.sequence = 0
        self.send_lock = asyncio.Lock()
        self.paused = False

    def wait(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        while True:
            try:
                return future.result(timeout=.2)
            except FutureTimeout:
                if self.stopped.is_set():
                    future.cancel()
                    raise RuntimeError("Browser playback stopped")

    async def send(self, event):
        async with self.send_lock:
            await self.socket.send_json(event)

    def on_event(self, event):
        if not self.stopped.is_set():
            self.wait(self.send(event))

    async def packet(self, audio=None):
        self.sequence += 1
        sequence = self.sequence
        acknowledged = asyncio.Event()
        self.pending[sequence] = acknowledged
        try:
            async with self.send_lock:
                if audio is None:
                    await self.socket.send_json({"type": "audio_end", "id": sequence})
                else:
                    await self.socket.send_bytes(struct.pack("<I", sequence) + audio)
            # A paused AudioContext cannot drain. Only time while playing counts.
            deadline = time.monotonic() + 60
            while not acknowledged.is_set():
                if self.stopped.is_set():
                    raise RuntimeError("Browser playback stopped")
                if self.paused:
                    deadline = time.monotonic() + 60
                elif time.monotonic() > deadline:
                    raise RuntimeError("The browser stopped acknowledging audio playback")
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(acknowledged.wait(), .2)
        finally:
            self.pending.pop(sequence, None)

    def acknowledge(self, sequence):
        event = self.pending.get(sequence)
        if event:
            event.set()

    def play_stream(self, chunks, sample_rate):
        self.wait(self.send({"type": "audio_start", "sample_rate": sample_rate}))
        try:
            for chunk in chunks:
                if self.stopped.is_set():
                    raise RuntimeError("Browser playback stopped")
                data = np.asarray(chunk, dtype="<f4").tobytes()
                self.wait(self.packet(data))
            # Finish the current segment before translating the next one.
            self.wait(self.packet())
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()


def create_app(runtime=None, output_root=Path("output/web"), *, preload=None):
    preload = runtime is None if preload is None else preload
    runtime = runtime if runtime is not None else ModelRuntime()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-asr")
    busy = asyncio.Lock()
    loading = False
    load_error = None

    async def prepare_in_background():
        nonlocal loading, load_error
        loading = True
        try:
            async with busy:
                await asyncio.get_running_loop().run_in_executor(executor, runtime.prepare)
        except Exception as exc:
            load_error = str(exc)
        finally:
            loading = False

    @asynccontextmanager
    async def lifespan(app):
        preparation = asyncio.create_task(prepare_in_background()) if preload else None
        try:
            yield
        finally:
            if preparation:
                await preparation
            await asyncio.get_running_loop().run_in_executor(executor, runtime.close)
            executor.shutdown(wait=True)

    app = FastAPI(title="Translate Video", lifespan=lifespan)
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    async def health():
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"status": "ok", "models_loaded": runtime.loaded, "busy": busy.locked(),
             "loading": loading, "load_error": load_error},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/example")
    def example():
        sample = Path(__file__).resolve().parents[2] / "examples" / "english_10s.wav"
        if not sample.is_file():
            raise HTTPException(404, "Example file is not installed")
        return FileResponse(sample, media_type="audio/wav")

    @app.websocket("/ws")
    async def stream(socket: WebSocket):
        origin = socket.headers.get("origin")
        if origin and urlsplit(origin).netloc != socket.headers.get("host"):
            await socket.close(code=1008)
            return
        await socket.accept()
        if busy.locked():
            await socket.send_json({"type": "error", "message": "Máy đang dịch file khác. Hãy thử lại khi phiên đó kết thúc."})
            await socket.close(code=1013)
            return
        async with busy:
            stopped = threading.Event()
            incoming = queue.Queue(maxsize=50)  # At most 5 seconds of input PCM.
            loop = asyncio.get_running_loop()
            bridge = BrowserBridge(socket, loop, stopped)
            worker = None
            receiver = None

            async def receive():
                ended = False
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    if message.get("bytes") is not None:
                        data = message["bytes"]
                        if ended or not data or len(data) > 6400 or len(data) % 4:
                            raise ValueError("Invalid input audio frame")
                        frame = np.frombuffer(data, dtype="<f4").copy()
                        if not np.isfinite(frame).all() or np.max(np.abs(frame)) > 1.01:
                            raise ValueError("Input audio must be finite PCM between -1 and 1")
                        try:
                            incoming.put_nowait(frame)
                        except queue.Full:
                            raise RuntimeError("Nhận dạng không theo kịp audio. Hãy dừng rồi thử file ngắn hơn.") from None
                    else:
                        command = json.loads(message.get("text", "{}"))
                        kind = command.get("type")
                        if kind == "ack" and isinstance(command.get("id"), int):
                            bridge.acknowledge(command["id"])
                        elif kind == "pause":
                            bridge.paused = True
                        elif kind == "resume":
                            bridge.paused = False
                        elif kind == "end" and not ended:
                            ended = True
                            try:
                                incoming.put_nowait(None)
                            except queue.Full:
                                raise RuntimeError("Input backlog exceeded five seconds") from None
                        elif kind == "stop":
                            raise WebSocketDisconnect()
                        else:
                            raise ValueError("Unknown or duplicate control message")

            def chunks():
                end = 0.0
                while not stopped.is_set():
                    try:
                        frame = incoming.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if frame is None:
                        return
                    end += len(frame) / 16000
                    if end > 600.1:
                        raise ValueError("Audio is limited to ten minutes per session")
                    yield frame, end
                    if not stopped.is_set():
                        bridge.wait(bridge.send({"type": "input_progress", "seconds": end}))

            async def watch(future):
                done, _ = await asyncio.wait([future, receiver], return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    await receiver  # Propagate disconnect / malformed input.
                return await asyncio.shield(future)

            try:
                await bridge.send({"type": "status", "message": "Đang nạp model… Lần đầu có thể mất một lúc."})
                receiver = asyncio.create_task(receive())
                worker = loop.run_in_executor(executor, runtime.prepare)
                await watch(worker)
                output = Path(output_root) / uuid.uuid4().hex
                pipeline = Pipeline(
                    runtime.asr, runtime.translator, runtime.tts, output,
                    player=bridge, on_event=bridge.on_event, stop_event=stopped,
                    max_segment_words=8, segment_holdback_words=3,
                    metadata={"interface": "web", "input_sample_rate": 16000,
                              "asr_chunk_seconds": .5, "max_segment_words": 8,
                              "segment_holdback_words": 3,
                              "tts": "pnnbao-ump/VieNeu-TTS-v3-Turbo"},
                )
                await bridge.send({"type": "ready", "sample_rate": runtime.tts.sample_rate})
                worker = loop.run_in_executor(executor, pipeline.run, chunks())
                await watch(worker)
                await bridge.send({"type": "done"})
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                with suppress(Exception):
                    await bridge.send({"type": "error", "message": str(exc)})
            finally:
                stopped.set()
                if receiver:
                    receiver.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await receiver
                if worker:
                    # Retain exclusive GPU ownership until the old pipeline exits.
                    with suppress(Exception):
                        await asyncio.shield(worker)
                with suppress(Exception):
                    await socket.close()

    return app


def main():
    parser = argparse.ArgumentParser(description="Open the local real-time dubbing web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    compiler = Path(__file__).resolve().parents[2] / "scripts" / "zig-cc"
    if compiler.is_file():
        os.environ.setdefault("CC", str(compiler))
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, ws_max_size=16384)


if __name__ == "__main__":
    main()
