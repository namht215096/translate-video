"""Exercise real WebSocket framing and pipeline lifecycle with small model fakes."""

import struct
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from translate_video.web import create_app


class ASR:
    def reset(self):
        pass

    def feed(self, audio):
        return "Hello.", True

    def finish(self):
        return "Hello."


class Translator:
    def translate(self, text):
        return "Xin chào."


class TTS:
    sample_rate = 48000

    def synthesize_stream(self, text):
        yield np.ones(480, dtype=np.float32) * .1
        yield np.ones(480, dtype=np.float32) * .2


class Runtime:
    loaded = False

    def prepare(self):
        self.asr, self.translator, self.tts = ASR(), Translator(), TTS()
        self.loaded = True

    def close(self):
        pass


def ready(socket):
    assert socket.receive_json()["type"] == "status"
    assert socket.receive_json()["type"] == "ready"


def drain(socket):
    events, audio = [], []
    while True:
        message = socket.receive()
        if message.get("bytes") is not None:
            data = message["bytes"]
            sequence, = struct.unpack("<I", data[:4])
            audio.extend(np.frombuffer(data[4:], dtype="<f4"))
            socket.send_json({"type": "ack", "id": sequence})
        else:
            import json
            event = json.loads(message["text"])
            events.append(event)
            if event["type"] == "audio_end":
                socket.send_json({"type": "ack", "id": event["id"]})
            if event["type"] in ("done", "error"):
                return events, audio


def test_audio_and_subtitles_stream_without_saving_audio(tmp_path):
    with TestClient(create_app(Runtime(), tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        with client.websocket_connect("/ws") as socket:
            ready(socket)
            socket.send_bytes(np.ones(1600, dtype="<f4").tobytes())
            socket.send_json({"type": "end"})
            events, audio = drain(socket)
            assert events[-1]["type"] == "done"
            assert next(e for e in events if e["type"] == "input_progress")["seconds"] == pytest.approx(.1)
            assert next(e for e in events if e["type"] == "translation")["translation"] == "Xin chào."
            assert len(audio) == 960
            assert audio[0] == pytest.approx(.1)
            assert audio[-1] == pytest.approx(.2)
        assert {path.name for path in next(tmp_path.iterdir()).iterdir()} == {"events.jsonl", "configuration.json"}


def test_browser_disconnect_releases_gpu_for_next_session(tmp_path):
    with TestClient(create_app(Runtime(), tmp_path)) as client:
        with client.websocket_connect("/ws") as socket:
            ready(socket)
            socket.send_bytes(np.ones(1600, dtype="<f4").tobytes())
            socket.send_json({"type": "end"})
            # Disconnect while the server is waiting for a playback acknowledgement.
            while "bytes" not in socket.receive():
                pass
        deadline = time.monotonic() + 3
        while client.get("/api/health").json()["busy"] and time.monotonic() < deadline:
            time.sleep(.02)
        assert not client.get("/api/health").json()["busy"]
        with client.websocket_connect("/ws") as socket:
            ready(socket)
            socket.send_json({"type": "end"})
            assert drain(socket)[0][-1]["type"] == "done"


def test_second_browser_is_rejected_while_session_is_active(tmp_path):
    with TestClient(create_app(Runtime(), tmp_path)) as client:
        with client.websocket_connect("/ws") as first:
            ready(first)
            with client.websocket_connect("/ws") as second:
                assert second.receive_json()["type"] == "error"


@pytest.mark.parametrize("data", [b"bad", np.array([np.nan], dtype="<f4").tobytes(), b"\0" * 6404])
def test_bad_audio_is_rejected(tmp_path, data):
    with TestClient(create_app(Runtime(), tmp_path)) as client:
        with client.websocket_connect("/ws") as socket:
            ready(socket)
            socket.send_bytes(data)
            assert socket.receive_json()["type"] == "error"


def test_foreign_origin_cannot_open_gpu_session(tmp_path):
    from starlette.websockets import WebSocketDisconnect

    with TestClient(create_app(Runtime(), tmp_path)) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://elsewhere.example"}):
                pass
