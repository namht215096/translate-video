# Changelog

This file records changes to the Python speech translation prototype.
The current development version is `0.1.0`; no versioned release has been published.

## [Unreleased]

### Added

- Local English → Vietnamese speech translation using
  `Qwen/Qwen3-ASR-1.7B` for streaming recognition through vLLM,
  `facebook/nllb-200-distilled-600M` for translation, and
  `facebook/mms-tts-vie` (VITS) for a fixed Vietnamese voice.
- Python 3.12 package and `translate-video` CLI, with language, model, device,
  segmentation, playback, and output-directory options.
- Audio-file replay in 100 ms frames, mono conversion, and resampling to 16 kHz.
  `--realtime` paces file input according to the source audio clock.
- Microphone input through `--mic`, with bounded capture buffering and explicit
  overflow errors. Microphone capture is implemented but has not been validated
  in the recorded experiments.
- Stable transcript segmentation, punctuation boundaries, a 24-word segment
  limit, silence finalization, and conservative `--final-only` mode.
- Separate ASR, translation/TTS, and audio-output execution lanes. Bounded text
  and audio queues preserve segment order while earlier speech is processed.
- Individual translated WAV files, a concatenated `translated.wav`, JSONL
  events, and a run configuration containing arguments and core library versions.
- Stage timings, first-waveform availability, and median/p95 reporting through
  `python -m translate_video.report`. Stage timings exclude listening and ASR
  stabilization; physical speaker latency remains unmeasured.
- Explicit errors for revisions to already committed text, worker failures,
  excessive backlog, translation truncation, and invalid NLLB language codes.
  Existing output directories are protected from accidental overwrite.
- Optional Zig compiler and `scripts/zig-cc` wrapper for machines without a
  system C compiler, plus compiler/GPU checks before model initialization.
- English setup instructions, feasibility notes, experiment reports, model
  revision records, and a snapshot of the libraries used in the experiments.
- A prepared ten-second English WAV excerpt and source provenance. Generated
  WAV files, model-related caches, local tools, and experiment output are ignored
  by Git.
- Matplotlib architecture diagrams in PNG and SVG, a reproducible drawing
  script, and the optional `plot` dependency group.
- Eight automated tests covering transcript stability, duplicate prevention,
  final flushing, segment limits, worker errors, audio draining, and resampling.

### Changed

- Increased the default maximum ASR window from 8 to 12 seconds after the shorter
  window split a negative phrase across translation segments in the sample.
- Reduced ASR context and batching budgets from 4096 to 1024 tokens, with one
  stream and eager inference, to fit the reserved memory budget on the test GPU.
- Revised the architecture diagram to use Vietnamese functional blocks,
  descriptions of each block, labeled data-flow arrows, two queues, three
  execution lanes, and separate logging connections.

### Fixed

- Enabled `--play` on Linux systems without PortAudio by selecting `pw-play`
  first, then `aplay`, and otherwise using `sounddevice` with PortAudio.
- Selected BF16 explicitly for Qwen inference to avoid vLLM 0.14 attempting a
  Hub dtype-metadata lookup with a local snapshot path and logging repeated
  repository-ID errors.
- Applied `fix_mistral_regex=True` to the vLLM runtime tokenizer, matching the
  correction used by Qwen's ASR wrapper. This does not eliminate every warning
  emitted by upstream processor initialization.
- Documented a disk-backed installation temporary directory after large CUDA
  wheels exhausted the machine's RAM-backed `/tmp` limit.

### Validation

- Eight automated tests passed, and `pip check` reported no broken requirements
  in the tested environment.
- Real model inference completed on an NVIDIA RTX 4070 SUPER with 12 GB VRAM,
  including both conservative and stable-prefix file replay.
- A ten-second file replay with `--realtime --play` completed with five segments
  submitted successfully to PipeWire. Its first waveform was ready after
  4.107 seconds of input processing, excluding model loading.
- Recorded outputs and stage statistics are documented in
  [the experiment report](docs/experiment.md). Short sample runs do not establish
  sustained live performance or physical speaker latency.

### Known limitations

- FastSpeech 2, the proposed alVoCat vocoder, and Nemotron are not integrated.
  MMS-TTS is the current TTS baseline.
- Silence detection uses RMS energy rather than a trained VAD. Forced ASR
  boundaries and independent translation segments can reduce translation quality.
- TTS generates full segments before playback. Longer translated speech can
  accumulate playback delay; automatic duration adaptation is not implemented.
- Voice cloning, speaker separation, video synchronization, and background-music
  preservation are not implemented.
- Long-session performance, microphone capture, and Vietnamese → English have
  not been validated in the recorded experiments. Upstream startup warnings
  may still appear.
