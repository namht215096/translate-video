# Windows setup notes

Tested on Windows 10, i5-12400F, 16 GB RAM, RTX 5060 Ti 16 GB (sm_120), Python 3.12.

- `scripts/setup_windows.ps1` builds `.venv` (PyTorch 2.11+cu130, community vLLM 0.29.0 Windows wheel, qwen-asr, vieneu).
- `scripts/run_windows.ps1 <translate-video args>` runs the CLI with a project-local CUDA 13 runtime
  (system CUDA 12.8 is untouched), caches under `.cache\`, and disables vLLM's FlashInfer sampler
  (it needs nvcc + MSVC to JIT).
- `scripts/benchmark_windows.ps1` runs both TTS backends in real time and writes
  `output\bench-<time>\` with per-run `events.jsonl`, `configuration.json` (includes `model_load_seconds`),
  `audio\segment_NNN.wav`, console logs, `process-times.txt`, and `comparison.md`.

Code deviations from the Linux setup (transformers 5 / vLLM 0.29 vs. the pinned 4.57.6 / 0.14):
- `vllm_compat.py`: shim so qwen-asr imports under transformers 5; register a renderer for the custom tokenizer mode.
- `models.py`: removed `swap_space`; VieNeu is warmed up at load time (its CUDA-graph capture on a worker
  thread otherwise collides with vLLM: "operation not permitted when stream is capturing").
- `cli.py`: no C-compiler check on Windows; `--save-audio`; model load times in `configuration.json`;
  `TRANSLATE_VIDEO_DEBUG=1` prints tracebacks.

First run after install triggers one-off Triton compilation (~50 s stall mid-run); later runs are cached.

## Desktop app (always-on-top window)

Start `TranslateVideo.exe` (build it with `scripts\build_exe.ps1`; it is a small launcher that runs
`.venv\Scripts\pythonw.exe -m translate_video.app`, so it works on this machine only) or `start_app.bat`.

1. **Khởi động** loads Qwen3-ASR, NLLB and VieNeu in a separate engine process (~1–1.5 min the first time;
   models stay loaded across Stop/Start). GUI never imports torch, so it never freezes.
2. It then opens the default output device twice: a WASAPI loopback capture and one persistent playback stream.
   A 0.3 s very quiet noise burst measures the playback→capture delay ("calibration").
3. Everything the PC plays (YouTube, music, ...) is captured, transcribed (English), translated and spoken in Vietnamese.
   Lower the YouTube volume yourself; the Vietnamese voice has its own volume slider (up to 200 %, clipped at full scale).

**Why subtraction instead of per-process capture:** Windows 10 build 19045 has no process loopback
(needs build 20438+), so the capture would otherwise hear our own TTS and translate it again. We know exactly
what we play, loopback is a digital copy (gain 1.0, delay ≈ 1–25 ms), so `echo.py` subtracts it
(measured 65 dB with MME, ~37 dB while other loud audio plays; cancellation is cleaned 0.4 s late so the delay can be
tracked on fresh audio). Safety nets: segments that are already Vietnamese are skipped (`language.py`);
until the delay is locked the capture is muted while TTS plays; the pipeline drops queued sentences older than 8 s.

Logs for review/comparison: `output/app-<time>/events.jsonl` + `configuration.json` (same format as the CLI, plus
`audio_stats` every 5 s: echo lock, delay, glitches). Compare with
`.venv\Scripts\python scripts\compare_runs.py output\app-* output\bench-*\vieneu`. Engine/vLLM output: `.cache\engine.log`.

Self-tests on real devices (audible, no human needed): `scripts\loopback_selftest.py`, `scripts\engine_selftest.py`,
`scripts\app_selftest.py`.

Known limits: English → Vietnamese only; only the default output device is captured (change it in Windows and restart);
ASR windows end after 0.6 s of silence, so continuous background music forces 12 s windows; the first run after install
stalls ~50 s once while Triton compiles kernels.

## Latency controls (stages, speed, segmentation)

- **Three independent stages.** `Pipeline` runs translation, TTS and playback as separate threads connected by queues
  (`translate -> tts_queue -> TTS -> play_queue -> playback`). A sentence goes to TTS the moment it is translated, and the
  translator immediately takes the next one; neither waits for the speaker.
- **Reading speed** (pitch preserved; streaming WSOLA in `stretch.py`, ~0.5 % of the audio duration in CPU time):
  base **x1.1** always, rising linearly to **x1.4** as the listener falls behind. The delay used is
  `time since the sentence was committed + seconds of speech still queued ahead of it`: x1.1 up to 1.5 s, x1.4 from 6 s
  (`speed_low_lag`/`speed_high_lag`). Per-sentence speed is logged as `speed` / `lag_at_tts_ms`.
- **No sentence is dropped** any more (`max_lag_seconds` exists but the app leaves it unset). Pressing Stop discards the
  queued speech immediately.
- **Segmentation** (`--segmentation` / "Cách cắt câu" in the window, applies when listening starts):
  `punctuation` (default) cuts at every `, . ; : ! ?`; `semantic` cuts at sentence ends, at a comma only after >= 8 words,
  and a long unpunctuated run before a clause opener (and, but, because, which, ...). Fewer, more complete pieces; it waits longer.
- CLI: `--speed 1.1 --max-speed 1.4 --segmentation punctuation`; in the window the "Tốc độ đọc cơ bản" slider changes the base speed live.
- New log events: `translated` (text as soon as it is translated) and, on every `segment`: `tts_queue_ms`, `speed`,
  `lag_at_tts_ms`, `commit_to_speaker_ms`, `playback_wait_ms`. The `segment` event is now written when the sentence has been played.
