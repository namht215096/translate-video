"""Summarize one or more run directories into a timing comparison table (Markdown + JSON)."""

import json
import statistics
import sys
from pathlib import Path


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def summarize(directory: Path):
    config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    segments = [e for e in events if e["type"] == "segment"]
    summary = next((e for e in events if e["type"] == "summary"), {})
    column = lambda name: [e[name] for e in segments if name in e]
    load = config.get("model_load_seconds", {})
    result = {
        "run": directory.name,
        "tts": config["arguments"]["tts_backend"],
        "realtime": config["arguments"]["realtime"],
        "model_load_s": {k: round(v, 2) for k, v in load.items()},
        "model_load_total_s": round(sum(load.values()), 2),
        "source_seconds": summary.get("source_seconds"),
        "wall_s": round(summary["wall_seconds"], 2) if summary else None,
        "segments": len(segments),
        "generated_audio_s": round(sum(column("audio_seconds")), 2),
        "first_audio_ready_s": round(min(column("audio_ready_elapsed_seconds")), 2) if segments else None,
    }
    for name in ("translation_ms", "tts_first_chunk_ms", "tts_ms", "commit_to_audio_ready_ms"):
        values = column(name)
        if values:
            result[name] = {"p50": round(statistics.median(values)), "p95": round(pct(values, 0.95)), "max": round(max(values))}
    asr = [e["asr_call_ms"] for e in events if e["type"] == "partial"]
    if asr:
        result["asr_call_ms"] = {"p50": round(statistics.median(asr)), "p95": round(pct(asr, 0.95)), "max": round(max(asr))}
    return result


def main():
    rows = [summarize(Path(arg)) for arg in sys.argv[1:]]
    metrics = ["translation_ms", "tts_first_chunk_ms", "tts_ms", "commit_to_audio_ready_ms", "asr_call_ms"]
    lines = ["| metric | " + " | ".join(r["run"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    simple = ["tts", "realtime", "source_seconds", "wall_s", "segments", "generated_audio_s",
              "first_audio_ready_s", "model_load_total_s"]
    for key in simple:
        lines.append(f"| {key} | " + " | ".join(str(r.get(key)) for r in rows) + " |")
    for part in ("asr", "nllb", "tts"):
        lines.append(f"| load_{part}_s | " + " | ".join(str(r["model_load_s"].get(part)) for r in rows) + " |")
    for key in metrics:
        for stat in ("p50", "p95", "max"):
            if any(key in r for r in rows):
                lines.append(f"| {key} {stat} | " + " | ".join(str(r.get(key, {}).get(stat, "-")) for r in rows) + " |")
    print("\n".join(lines))
    print("\n" + json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
