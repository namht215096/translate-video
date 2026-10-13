"""Command line entry point."""

import argparse
import os
from pathlib import Path
import shutil
import sys


def parser():
    result = argparse.ArgumentParser(description="Experimental streaming speech translation")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="WAV/FLAC file replayed in 100 ms chunks")
    source.add_argument("--mic", action="store_true", help="Capture the microphone; Ctrl+C drains pending speech")
    result.add_argument("--mic-device", help="PortAudio input device name")
    result.add_argument("--realtime", action="store_true", help="Pace file input according to its audio clock")
    result.add_argument("--play", action="store_true", help="Play translated speech through the default output device")
    result.add_argument("--output", type=Path, required=True, help="New directory for JSON logs and configuration; audio is not saved")
    result.add_argument("--save-audio", action="store_true", help="Also write each translated segment as audio/segment_NNN.wav in --output")
    result.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    result.add_argument("--source-language", default="English")
    result.add_argument("--source-code", default="eng_Latn", help="NLLB source language code")
    result.add_argument("--target-code", default="vie_Latn", help="NLLB target language code")
    result.add_argument("--tts-backend", choices=["vieneu", "mms"], default="vieneu")
    result.add_argument("--tts-model", help="Override the checkpoint for the selected TTS backend")
    result.add_argument("--tts-voice", help="VieNeu preset voice; omitted uses the SDK default")
    result.add_argument("--model-device", default="cuda:0", help="Device for NLLB and TTS; ASR requires CUDA")
    result.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    result.add_argument("--chunk-seconds", type=float, default=1.0)
    result.add_argument("--silence-seconds", type=float, default=0.6)
    result.add_argument("--max-window-seconds", type=float, default=12.0)
    result.add_argument("--rms-threshold", type=float, default=0.008)
    result.add_argument("--speed", type=float, default=1.1, help="Base speaking speed (pitch preserved); 1.0 disables stretching")
    result.add_argument("--max-speed", type=float, default=1.4, help="Speed reached when the listener falls behind; set equal to --speed to disable")
    result.add_argument("--segmentation", choices=["punctuation", "semantic"], default="punctuation",
                        help="punctuation: cut at every comma/period (lowest delay); semantic: keep clauses together")
    result.add_argument("--final-only", action="store_true", help="Commit text only when each ASR window finishes")
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    args.tts_model = args.tts_model or (
        "pnnbao-ump/VieNeu-TTS-v3-Turbo" if args.tts_backend == "vieneu"
        else "facebook/mms-tts-vie"
    )
    if args.output.exists():
        arguments.error("--output already exists; choose a new directory")
    if args.input and not args.input.is_file():
        arguments.error("--input must be an existing audio file")
    if not 0 < args.gpu_memory_utilization < 1:
        arguments.error("--gpu-memory-utilization must be between 0 and 1")
    if min(args.chunk_seconds, args.silence_seconds, args.max_window_seconds) <= 0:
        arguments.error("time settings must be positive")
    if args.max_window_seconds < args.chunk_seconds:
        arguments.error("--max-window-seconds must be at least --chunk-seconds")
    if args.speed < 1 or args.max_speed < args.speed or args.max_speed > 2:
        arguments.error("--speed must be >= 1 and --max-speed between --speed and 2")
    if args.rms_threshold < 0:
        arguments.error("--rms-threshold must be nonnegative")
    try:
        import torch
        from importlib.metadata import version
        from .audio import file_chunks, microphone_chunks
        from .models import MMSTTS, VieNeuTTS, NLLBTranslator, QwenASR
        from .pipeline import Pipeline

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen streaming requires a compatible CUDA GPU")
        if sys.platform != "win32" and not os.environ.get("CC") and not any(shutil.which(name) for name in ("cc", "gcc", "clang")):
            raise RuntimeError("A C compiler is required by Triton. Install GCC or use the optional Zig compiler described in README.md")
        import time

        load_seconds = {}
        mark = time.monotonic()
        print("Loading Qwen ASR...", flush=True)
        asr = QwenASR(args.asr_model, args.source_language, args.chunk_seconds, args.gpu_memory_utilization)
        load_seconds["asr"] = time.monotonic() - mark
        print(f"Loading NLLB and {args.tts_backend} TTS...", flush=True)
        mark = time.monotonic()
        translator = NLLBTranslator(args.source_code, args.target_code, args.model_device)
        load_seconds["nllb"] = time.monotonic() - mark
        mark = time.monotonic()
        tts = (VieNeuTTS(args.tts_model, args.model_device, args.tts_voice)
               if args.tts_backend == "vieneu" else MMSTTS(args.tts_model, args.model_device))
        load_seconds["tts"] = time.monotonic() - mark
        chunks = (microphone_chunks(args.mic_device) if args.mic
                  else file_chunks(args.input, realtime=args.realtime))
        pipeline = Pipeline(
            asr, translator, tts, args.output, play=args.play,
            silence_seconds=args.silence_seconds,
            max_window_seconds=args.max_window_seconds,
            rms_threshold=args.rms_threshold, final_only=args.final_only,
            save_audio=args.save_audio, speed=args.speed, max_speed=args.max_speed,
            segmentation=args.segmentation,
            metadata={"arguments": vars(args), "model_load_seconds": load_seconds, "packages": {
                name: version(name) for name in ["qwen-asr", "vllm", "torch", "transformers"]
                + (["vieneu", "sea-g2p"] if args.tts_backend == "vieneu" else [])
            }, "gpu": torch.cuda.get_device_name(0), "tts": {
                "sample_rate": tts.sample_rate,
                "voice": args.tts_voice or getattr(tts.model, "_default_voice", None),
            }},
        )
        try:
            pipeline.run(chunks)
        finally:
            if hasattr(tts, "close"):
                tts.close()
    except Exception as exc:
        if os.environ.get("TRANSLATE_VIDEO_DEBUG"):
            import traceback
            traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
