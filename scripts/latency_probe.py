"""True end-to-end latency: end of each spoken sentence -> first Vietnamese sample handed to the speaker.

Ground truth comes from a clip assembled from separately synthesized sentences (exact start/end times).
The clip is replayed in real time through the real models and Pipeline. Playback is simulated by
blocking for each segment's audio duration (what a real speaker stream does), so a backlog builds exactly
as in the app. No audio devices are used.

  .cache\\latency\\ holds s00.wav.. and sentences.json (see docs/windows-setup.md).
  scripts\\run_windows.ps1-style environment is applied here. Run:
  .venv\\Scripts\\python scripts\\latency_probe.py [--gap 2.5 --gap 0.35] [--out output\\latency-<time>]
"""

import argparse
from datetime import datetime
import difflib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache" / "latency"


class RealtimePlayer:
    """Blocks for the audio's duration, like a speaker stream that is drained at the sample rate."""

    name = "simulated"

    def play(self, audio, sample_rate):
        time.sleep(len(audio) / sample_rate)

    def play_stream(self, chunks, sample_rate):
        start, total = time.monotonic(), 0.0
        try:
            for chunk in chunks:
                total += len(chunk) / sample_rate
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()
        time.sleep(max(0.0, start + total - time.monotonic()))


def build_clip(gap: float, lead: float = 1.0):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    sentences = json.loads((DATA / "sentences.json").read_text(encoding="utf-8-sig"))
    pieces, spans, cursor = [np.zeros(int(lead * 16000), np.float32)], [], lead
    for index, text in enumerate(sentences):
        audio, rate = sf.read(DATA / f"s{index:02d}.wav", dtype="float32")
        audio = audio if audio.ndim == 1 else audio.mean(axis=1)
        if rate != 16000:
            audio = resample_poly(audio, 16000, rate).astype(np.float32)
        voiced = np.flatnonzero(np.abs(audio) > 0.01)          # trim SAPI's silent head/tail
        audio = audio[voiced[0]:voiced[-1] + 1]
        spans.append({"text": text, "start": cursor, "end": cursor + len(audio) / 16000})
        pieces += [audio, np.zeros(int(gap * 16000), np.float32)]
        cursor += len(audio) / 16000 + gap
    return np.concatenate(pieces), spans


def run(models, clip, spans, output: Path, config):
    import numpy as np
    import soundfile as sf

    from translate_video.audio import file_chunks
    from translate_video.pipeline import Pipeline

    wav = output.parent / f"{output.name}.wav"
    sf.write(wav, clip, 16000)
    models["asr"].reset()
    Pipeline(models["asr"], models["translator"], models["tts"], output, player=RealtimePlayer(),
             on_revision="reset", speed=config["speed"], max_speed=config["max_speed"],
             segmentation=config["segmentation"],
             metadata={"arguments": {"tts_backend": "vieneu", "realtime": True, "probe": True, **config}}
             ).run(file_chunks(wav, realtime=True))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    segments = [e for e in events if e["type"] == "segment"]
    dropped = [e for e in events if e["type"] == "dropped"]
    return spans, segments, dropped


def match(spans, segments, dropped):
    """Attach each segment to the sentence it belongs to.

    A fragment ("This video is.") must not be pulled to a different sentence by edit distance, so only sentences
    whose speech had already started when the segment was committed (and ended < 20 s earlier) are candidates,
    and the score is how much of the fragment appears inside the sentence.
    """
    rows = {i: {"span": s, "segments": [], "dropped": []} for i, s in enumerate(spans)}
    for kind, items in (("segments", segments), ("dropped", dropped)):
        for item in items:
            text = item["source"].lower()
            committed = item.get("commit_elapsed_seconds")
            candidates = [i for i, span in enumerate(spans)
                          if committed is None or span["start"] <= committed <= span["end"] + 20] or list(range(len(spans)))

            def score(i):
                other = spans[i]["text"].lower()
                longest = difflib.SequenceMatcher(None, text, other, autojunk=False).find_longest_match(0, len(text), 0, len(other))
                return longest.size / max(len(text), 1), -abs((committed or 0) - spans[i]["end"])

            best = max(candidates, key=score)
            rows[best][kind].append(item)
    return rows


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def report(name, rows):
    table, e2e = [], []
    parts = {k: [] for k in ("stabilize", "translate_queue", "translate", "tts_queue", "tts_first", "speaker_wait")}
    speeds = []
    for i, row in rows.items():
        end = row["span"]["end"]
        if not row["segments"]:
            table.append((i, "-", "-", "-", "-", "-", "-", "-", "-", "LOST" if row["dropped"] else "MISSED"))
            continue
        # a sentence may be split into several segments: the listener waits for the LAST one
        last = max(row["segments"], key=lambda s: s["commit_elapsed_seconds"])
        values = {
            "stabilize": last["commit_elapsed_seconds"] - end,
            "translate_queue": last["translation_queue_ms"] / 1000,
            "translate": last["translation_ms"] / 1000,
            "tts_queue": last["tts_queue_ms"] / 1000,
            "tts_first": last["tts_first_chunk_ms"] / 1000,
            "speaker_wait": last["playback_wait_ms"] / 1000,
        }
        total = last["playback_submit_elapsed_seconds"] - end
        e2e.append(total)
        speeds.append(last["speed"])
        for key, value in values.items():
            parts[key].append(value)
        table.append((i, f"{values['stabilize']:.1f}", f"{values['translate_queue']:.1f}", f"{values['translate']:.2f}",
                      f"{values['tts_queue']:.2f}", f"{values['tts_first']:.2f}", f"{values['speaker_wait']:.1f}",
                      f"{last['speed']:.2f}", f"{total:.1f}", f"{len(row['segments'])} seg"))
    print(f"\n### {name}")
    print("| sentence | wait for commit | translate queue | translate | TTS queue | TTS 1st chunk | speaker wait | speed | speech end -> speaker (s) | note |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in table:
        print("| " + " | ".join(str(x) for x in row) + " |")
    summary = {"name": name, "sentences": len(rows), "heard": len(e2e), "lost_or_missed": len(rows) - len(e2e)}
    if e2e:
        summary.update({"e2e_p50": statistics.median(e2e), "e2e_p95": pct(e2e, .95), "e2e_max": max(e2e),
                        "speed_p50": statistics.median(speeds), "speed_max": max(speeds)})
        for key, values in parts.items():
            summary[f"{key}_p50"] = statistics.median(values)
    print("\n" + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()}))
    return summary


def recompute(directory: Path):
    """Re-score saved runs (e.g. after a matcher fix) without loading any model."""
    summaries = []
    for run_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        gap = float(run_dir.name.split("-")[0].removeprefix("gap").removesuffix("s"))
        _, spans = build_clip(gap)
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        segments = [e for e in events if e["type"] == "segment"]
        dropped = [e for e in events if e["type"] == "dropped"]
        summaries.append(report(run_dir.name, match(spans, segments, dropped)))
    (directory / "latency_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", type=Path, help="re-score the runs saved in this directory and exit")
    parser.add_argument("--gap", type=float, action="append", help="silence between sentences; repeatable")
    parser.add_argument("--config", action="append",
                        help="name:speed:max_speed:segmentation (repeatable); default compares three setups")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / datetime.now().strftime("latency-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if args.recompute:
        return recompute(args.recompute)
    gaps = args.gap or [2.5, 0.35]
    specs = args.config or ["serial-1.0x:1.0:1.0:punctuation", "default-1.1-1.4x:1.1:1.4:punctuation",
                            "semantic-1.1-1.4x:1.1:1.4:semantic"]
    configs = []
    for spec in specs:
        name, speed, max_speed, segmentation = spec.split(":")
        configs.append({"name": name, "speed": float(speed), "max_speed": float(max_speed), "segmentation": segmentation})

    from translate_video.app import windows_env
    windows_env.configure(ROOT)
    from translate_video.app.engine import Settings
    from translate_video.models import NLLBTranslator, QwenASR, VieNeuTTS

    settings = Settings()
    print("loading models...", flush=True)
    models = {
        "asr": QwenASR("Qwen/Qwen3-ASR-1.7B", "English", settings.chunk_seconds, settings.gpu_memory_utilization),
        "translator": NLLBTranslator("eng_Latn", "vie_Latn", settings.model_device),
        "tts": VieNeuTTS("pnnbao-ump/VieNeu-TTS-v3-Turbo", settings.model_device, None),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for gap in gaps:
        for config in configs:
            clip, spans = build_clip(gap)
            name = f"gap{gap}s-{config['name']}"
            spans, segments, dropped = run(models, clip, spans, args.out / name, config)
            summaries.append(report(f"{name}: sentences {gap}s apart, {config['name']}", match(spans, segments, dropped)))
    (args.out / "latency_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\nwritten to", args.out)


if __name__ == "__main__":
    sys.exit(main())
