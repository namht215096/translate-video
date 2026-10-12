"""Compare TTS on identical texts, without playback or saved waveforms."""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from translate_video.models import MMSTTS, VieNeuTTS


TEXTS = [
    "Xin chào.",
    "Ồ, đúng rồi.",
    "Hôm nay trời đẹp, chúng ta cùng đi dạo nhé.",
    "Anh ta thậm chí còn không lớn đến thế khi tôi bắt đầu lắng nghe anh ta,",
    "nhưng và âm nhạc solo của anh ấy không làm quá nhiều.",
    "Đây là hệ thống dịch giọng nói từ tiếng Anh sang tiếng Việt trong thời gian thực. "
    "Âm thanh được nhận dạng, dịch thành văn bản và đọc bằng giọng nói tổng hợp "
    "để người nghe có thể theo dõi nội dung.",
]


def stats(values):
    ordered = sorted(values)
    return {"p50": float(np.median(ordered)),
            "p95": ordered[math.ceil(len(ordered) * .95) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mms", "vieneu"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--voice")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.warmups < 1:
        parser.error("Choose a new output file and positive repeat/warmup counts")
    torch.set_num_threads(2)
    gpu = args.device.startswith("cuda")

    def sync():
        if gpu:
            torch.cuda.synchronize(args.device)

    model_id = ("facebook/mms-tts-vie" if args.backend == "mms"
                else "pnnbao-ump/VieNeu-TTS-v3-Turbo")
    loading = time.perf_counter()
    tts = (MMSTTS(model_id, args.device) if args.backend == "mms"
           else VieNeuTTS(model_id, args.device, args.voice))
    sync()
    load_ms = (time.perf_counter() - loading) * 1000

    def measure(text, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sync()
        started = time.perf_counter()
        chunks = (tts.synthesize_stream(text) if args.backend == "vieneu"
                  else iter([tts.synthesize(text)]))
        first = None
        samples = count = 0
        for chunk in chunks:
            if chunk.ndim != 1 or not np.isfinite(chunk).all():
                raise RuntimeError("Invalid TTS waveform")
            if len(chunk):
                first = first or time.perf_counter()
                samples += len(chunk)
                count += 1
        sync()
        elapsed = time.perf_counter() - started
        if first is None or samples == 0:
            raise RuntimeError("TTS produced no audio")
        seconds = samples / tts.sample_rate
        return {"text": text, "seed": seed, "first_audio_ms": (first - started) * 1000,
                "total_ms": elapsed * 1000, "audio_seconds": seconds,
                "rtf": elapsed / seconds, "chunks": count}

    try:
        # Includes graph/scheduler initialization, intentionally outside warm results.
        cold = measure(TEXTS[2], 42)
        for i in range(args.warmups):
            measure(TEXTS[2], 100 + i)
        schedule = list(range(len(TEXTS))) * args.repeats
        random.Random(42).shuffle(schedule)
        rows = []
        for i, index in enumerate(schedule):
            row = measure(TEXTS[index], 1000 + i)
            row["text_index"] = index
            rows.append(row)
            print(f"{args.backend} {i+1}/{len(schedule)}: first={row['first_audio_ms']:.1f}ms "
                  f"total={row['total_ms']:.1f}ms RTF={row['rtf']:.3f}", flush=True)
        from huggingface_hub.constants import HF_HUB_CACHE
        revisions = {}
        for repo in [model_id] + (["OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"]
                                 if args.backend == "vieneu" else []):
            ref = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs/main"
            revisions[repo] = ref.read_text().strip() if ref.exists() else None
        result = {
            "backend": args.backend, "model": model_id, "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device) if gpu else None,
            "sample_rate": tts.sample_rate,
            "voice": getattr(getattr(tts, "model", None), "_default_voice", None)
                     if args.voice is None else args.voice,
            "max_streams": 1 if args.backend == "vieneu" else None,
            "load_ms": load_ms, "cold_request": cold, "warmups": args.warmups,
            "repeats": args.repeats, "revisions": revisions,
            "packages": {name: version(name) for name in
                         ["torch", "transformers", "vieneu", "sea-g2p", "onnxruntime"]},
            "summary": {key: stats([row[key] for row in rows]) for key in
                        ["first_audio_ms", "total_ms", "audio_seconds", "rtf"]},
            "per_text": [{"text": text, **{key: stats([row[key] for row in rows
                                                       if row["text_index"] == i])
                           for key in ["first_audio_ms", "total_ms", "audio_seconds", "rtf"]}}
                         for i, text in enumerate(TEXTS)],
            "measurements": rows,
            "method": "Sequential requests, seed matched by request, 2 CPU threads, "
                      "CUDA synchronized at boundaries, no playback, no ASR/MT, "
                      "no audio saved. MMS first audio is its entire waveform; "
                      "VieNeu uses infer_stream with watermark disabled. "
                      "Different native rates and generated durations; no quality scoring.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    finally:
        if hasattr(tts, "close"):
            tts.close()


if __name__ == "__main__":
    main()
