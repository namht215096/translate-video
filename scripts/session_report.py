"""Summarize recent app sessions (output/app-*) so they can be compared and judged.

  .venv\\Scripts\\python scripts\\session_report.py            # last 6 sessions
  .venv\\Scripts\\python scripts\\session_report.py -n 3 --out report.md
  .venv\\Scripts\\python scripts\\session_report.py output\\app-20261014-004148 ...

"wait to commit" is estimated from the ASR partials: the time between the ASR first producing the end of a
piece and the piece being committed (values > 10 s are matching artefacts and are ignored).
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from translate_video.language import contains_cjk  # noqa: E402
from translate_video.segmentation import weight, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PUNCT = re.compile(r"[，。！？、；：,.!?;:\s]+")


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else None


def fmt(value, digits=1, unit=""):
    return "-" if value is None else f"{value:.{digits}f}{unit}"


def units(text):
    return sum(weight(token) for _, token in tokenize(text))


def load(directory: Path):
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    return events, config


def summarize(directory: Path):
    events, config = load(directory)
    arguments = config.get("arguments", {})
    segments = [e for e in events if e["type"] == "segment"]
    partials = [e for e in events if e["type"] == "partial"]
    stats = [e for e in events if e["type"] == "audio_stats"]
    summary = next((e for e in events if e["type"] == "summary"), None)
    count = lambda kind: sum(e["type"] == kind for e in events)
    duration = summary["wall_seconds"] if summary else (stats[-1]["elapsed_seconds"] if stats else 0.0)
    column = lambda key: [e[key] for e in segments if key in e]

    waits = []
    for e in segments:
        tail = PUNCT.sub("", e["source"])[-5:]
        if len(tail) < 3:
            continue
        seen = next((p["source_observed_end_seconds"] for p in partials
                     if tail in PUNCT.sub("", p["text"]) and p["source_observed_end_seconds"] <= e["commit_elapsed_seconds"] + 1), None)
        if seen is not None and 0 <= e["commit_elapsed_seconds"] - seen <= 10:
            waits.append(e["commit_elapsed_seconds"] - seen)

    to_speaker = [v / 1000 for v in column("commit_to_speaker_ms")]
    speeds = column("speed")
    cjk = [e for e in segments if contains_cjk(e["source"])]
    audio = [e["audio_seconds"] * e.get("speed", 1.0) for e in segments]
    raw_speech = sum(audio)
    return {
        "name": directory.name,
        "language": f"{len(cjk) / len(segments):.0%} zh" if segments and cjk else "en",
        "duration": duration,
        "clean_end": summary is not None,
        "segments": len(segments),
        "settings": f"{arguments.get('segmentation', '-')}, x{arguments.get('speed', '-')}->{arguments.get('max_speed', '-')}",
        "wait_commit_p50": statistics.median(waits) if waits else None,
        "wait_commit_p95": pct(waits, .95),
        "to_speaker_p50": statistics.median(to_speaker) if to_speaker else None,
        "to_speaker_p95": pct(to_speaker, .95),
        "to_speaker_max": max(to_speaker) if to_speaker else None,
        "est_total_p50": (statistics.median(waits) if waits else 0) + (statistics.median(to_speaker) if to_speaker else 0),
        "translate_ms_p50": statistics.median(column("translation_ms")) if segments else None,
        "tts_first_ms_p50": statistics.median(column("tts_first_chunk_ms")) if column("tts_first_chunk_ms") else None,
        "speed_mean": statistics.mean(speeds) if speeds else None,
        "speed_max": max(speeds) if speeds else None,
        "speech_share": raw_speech / duration if duration else None,
        "copied": sum(1 for e in segments if contains_cjk(e["translation"]) and contains_cjk(e["source"])),
        "silent": sum(1 for e in segments if e["audio_seconds"] < 0.1 and units(e["source"]) >= 1),
        "tiny": sum(1 for e in segments if units(e["source"]) < 2),
        "cut_mid": sum(1 for e in segments if re.search(r"[的了不是还和与在对把被位于而但]。$", e["source"])),
        "resets": count("revision_reset"), "dropped": count("dropped"), "skipped": count("skipped"),
        "failed": count("translation_failed") + count("tts_failed"),
        "transcript": [(e["index"], e["commit_to_speaker_ms"] / 1000 if "commit_to_speaker_ms" in e else None,
                        e.get("speed"), e["source"], e["translation"]) for e in segments],
    }


def render(rows, transcripts):
    lines = ["| session | lang | length | ended | segs | cuts / speed | wait to commit p50/p95 | commit->speaker p50/p95/max | est. total p50 | speed mean/max | TTS speech share |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['language']} | {r['duration']:.0f}s | {'clean' if r['clean_end'] else 'no summary (crash/killed/still running)'} | {r['segments']} | {r['settings']} | "
            f"{fmt(r['wait_commit_p50'])} / {fmt(r['wait_commit_p95'])} s | {fmt(r['to_speaker_p50'])} / {fmt(r['to_speaker_p95'])} / {fmt(r['to_speaker_max'])} s | "
            f"{fmt(r['est_total_p50'])} s | {fmt(r['speed_mean'], 2)} / {fmt(r['speed_max'], 2)} | {fmt(100 * r['speech_share'], 0) if r['speech_share'] is not None else '-'} % |")
    lines += ["", "| session | model latency (translate / TTS 1st chunk p50) | translation copied unchanged | no audio produced | tiny pieces | cut mid-sentence (…的。) | ASR text revised | dropped | skipped (already Vietnamese) | errors tolerated |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {fmt(r['translate_ms_p50'], 0)} / {fmt(r['tts_first_ms_p50'], 0)} ms | {r['copied']} | {r['silent']} | {r['tiny']} | {r['cut_mid']} | {r['resets']} | {r['dropped']} | {r['skipped']} | {r['failed']} |")
    if transcripts:
        for r in rows:
            lines += ["", f"### {r['name']}", "", "| # | speaker +s | x | source | translation |", "|---|---|---|---|---|"]
            lines += [f"| {i} | {fmt(lag)} | {fmt(sp, 2)} | {src} | {tr} |" for i, lag, sp, src, tr in r["transcript"]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="*", type=Path)
    parser.add_argument("-n", type=int, default=6, help="how many of the latest app-* sessions when none are listed")
    parser.add_argument("--transcripts", action="store_true", help="also list every source/translation pair")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    directories = args.sessions or sorted((ROOT / "output").glob("app-*"), key=lambda p: p.stat().st_mtime)[-args.n:]
    rows = [summarize(d) for d in directories if (d / "events.jsonl").exists()]
    text = render(rows, args.transcripts)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
