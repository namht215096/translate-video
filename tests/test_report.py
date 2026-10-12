import json
import sys

import pytest

from translate_video.report import main


@pytest.mark.parametrize("extra", [{}, {"playback_ms": 100}, {"audio_queue_ms": 20}])
def test_report_supports_direct_playback_and_historical_logs(tmp_path, monkeypatch, capsys, extra):
    event = {
        "type": "segment", "audio_seconds": 0.1,
        "translation_queue_ms": 1, "translation_ms": 2, "tts_ms": 3,
        "commit_to_audio_ready_ms": 6, **extra,
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event) + "\n" + json.dumps({"type": "summary", "segments": 1}))
    monkeypatch.setattr(sys, "argv", ["report", str(path)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["completed"]
    assert result["timings_ms"]["tts_ms"]["p50"] == 3
    for metric in ["playback_ms", "audio_queue_ms"]:
        assert (metric in result["timings_ms"]) == (metric in extra)


def test_streaming_report_uses_first_chunk_time(tmp_path, monkeypatch, capsys):
    event = {
        "type": "segment", "audio_seconds": 1,
        "tts_first_chunk_ms": 40, "tts_stream_elapsed_ms": 300,
        "first_audio_ready_elapsed_seconds": 4.0,
        "audio_ready_elapsed_seconds": 4.26,
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event))
    monkeypatch.setattr(sys, "argv", ["report", str(path)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["first_audio_ready_seconds_from_input_start"] == 4.0
    assert result["timings_ms"]["tts_first_chunk_ms"]["p50"] == 40
    assert "tts_ms" not in result["timings_ms"]
