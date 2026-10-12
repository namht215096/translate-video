"""Summarize recorded stage timings without claiming end-to-end latency."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser(description="Summarize a translate-video events.jsonl file")
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = [event for event in events if event["type"] == "segment"]
    if not segments:
        parser.error("No completed segments were recorded")
    summaries = [event for event in events if event["type"] == "summary"]
    report = {
        "completed": bool(summaries),
        "segments": len(segments),
        "generated_audio_seconds": sum(event["audio_seconds"] for event in segments),
        "timings_ms": {},
        "note": "Stage timings exclude listening and ASR stabilization; these are not end-to-end latency.",
    }
    for field in ["translation_queue_ms", "translation_ms", "tts_ms", "audio_queue_ms", "commit_to_audio_ready_ms"]:
        values = [event[field] for event in segments]
        report["timings_ms"][field] = {"p50": round(percentile(values, 0.5), 2),
                                         "p95": round(percentile(values, 0.95), 2)}
    asr_calls = [event["asr_call_ms"] for event in events if event["type"] == "partial"]
    if asr_calls:
        report["timings_ms"]["asr_call_ms"] = {
            "p50": round(percentile(asr_calls, 0.5), 2),
            "p95": round(percentile(asr_calls, 0.95), 2),
        }
    if summaries:
        report["run"] = summaries[-1]
    if "audio_ready_elapsed_seconds" in segments[0]:
        report["first_audio_ready_seconds_from_input_start"] = round(segments[0]["audio_ready_elapsed_seconds"], 3)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
