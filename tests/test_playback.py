from types import SimpleNamespace

import numpy as np
import pytest

from translate_video.audio import AudioPlayer


@pytest.mark.parametrize("backend", ["pw-play", "aplay"])
def test_native_playback_pipes_pcm_without_audio_files(backend, monkeypatch):
    monkeypatch.setattr("translate_video.audio.shutil.which",
                        lambda name: f"/usr/bin/{name}" if name == backend else None)
    sent = []

    def run(command, **kwargs):
        sent.append((command, kwargs["input"]))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("translate_video.audio.subprocess.run", run)
    AudioPlayer().play(np.array([2.0, -2.0, 0.5]), 16000)
    command, pcm = sent[0]
    assert command[0].endswith(backend) and command[-1] == "-"
    assert "16000" in command
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767, 16383]


def test_native_player_error_is_reported(monkeypatch):
    monkeypatch.setattr("translate_video.audio.shutil.which", lambda name: "/usr/bin/pw-play")
    monkeypatch.setattr("translate_video.audio.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr=b"device unavailable"))
    with pytest.raises(RuntimeError, match="device unavailable"):
        AudioPlayer().play(np.zeros(1600), 16000)


def test_stream_uses_one_process_for_all_chunks(monkeypatch):
    sent = []

    class Pipe:
        closed = False

        def write(self, data):
            sent.append(data)

        def flush(self):
            pass

        def close(self):
            self.closed = True

    process = SimpleNamespace(stdin=Pipe(), wait=lambda: 0)
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr("translate_video.audio.shutil.which", lambda name: "/usr/bin/pw-play")
    monkeypatch.setattr("translate_video.audio.subprocess.Popen", popen)
    AudioPlayer().play_stream(iter([np.array([.5]), np.array([-.5])]), 48000)
    assert len(commands) == 1 and "48000" in commands[0]
    assert np.frombuffer(b"".join(sent), dtype="<i2").tolist() == [16383, -16383]
    assert process.stdin.closed


def test_stream_failure_closes_generator_and_stops_player(monkeypatch):
    stopped = []

    class Pipe:
        closed = False

        def write(self, data):
            raise BrokenPipeError("player stopped")

        def close(self):
            self.closed = True

    process = SimpleNamespace(stdin=Pipe(), poll=lambda: None,
                              terminate=lambda: stopped.append("terminate"),
                              wait=lambda: stopped.append("wait"))

    def chunks():
        try:
            yield np.zeros(100)
        finally:
            stopped.append("generator closed")

    monkeypatch.setattr("translate_video.audio.shutil.which", lambda name: "/usr/bin/pw-play")
    monkeypatch.setattr("translate_video.audio.subprocess.Popen", lambda *a, **kw: process)
    with pytest.raises(BrokenPipeError):
        AudioPlayer().play_stream(chunks(), 48000)
    assert stopped == ["terminate", "wait", "generator closed"]
    assert process.stdin.closed
