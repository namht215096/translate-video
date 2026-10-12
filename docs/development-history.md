# Development history

Detailed implementation history for the Python speech translation prototype.
See [the concise changelog](../CHANGELOG.md) for the current changes.
The current project version is `0.0.2`.

## [0.0.2] - 2026-10-12

### Added

- VieNeu-TTS-v3-Turbo integration through `vieneu==3.8.3`, selected by default.
  `--tts-backend mms` retains the previous baseline; `--tts-voice` selects a
  VieNeu preset. GPU inference is verified with the existing PyTorch 2.9.1.
- Incremental 48 kHz PCM playback through one player per utterance, without an
  application audio queue or saved waveforms. Logs distinguish first-chunk
  latency from streaming elapsed time, which overlaps playback.
- A reproducible TTS benchmark with six matched texts, 30 measured requests
  per backend, warmup separation, native audio duration, RTF, and raw JSON
  measurements. See `docs/tts-benchmark.md` for measured results and limitations.

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
- Separate ASR and translation/TTS/playback execution lanes. A bounded text queue
  preserves segment order while ASR continues processing input.
- JSONL events and a run configuration containing arguments and core library
  versions. Waveforms are played directly from memory when `--play` is enabled.
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
- Automated tests covering transcript stability, duplicate prevention, final
  flushing, segment limits, worker errors, direct PCM playback, the absence of
  saved audio, report compatibility, and resampling.

### Changed

- Increased the default maximum ASR window from 8 to 12 seconds after the shorter
  window split a negative phrase across translation segments in the sample.
- Reduced ASR context and batching budgets from 4096 to 1024 tokens, with one
  stream and eager inference, to fit the reserved memory budget on the test GPU.
- Revised the architecture diagram to use Vietnamese functional blocks,
  descriptions of each block, labeled data-flow arrows, the text queue, two
  execution lanes, and separate logging connections.
- Translation, TTS, and playback now run sequentially in one worker. Each segment
  finishes playback before the next segment is translated and synthesized.
- `--output` now selects a directory containing only JSON logs and configuration.
- Reporting now includes optional blocking playback duration and remains
  compatible with historical logs containing audio-queue timing.

### Removed

- The audio queue and separate audio-output worker.
- Saving individual or concatenated translated WAV files, including temporary
  audio files for playback. PipeWire/ALSA receive in-memory PCM through stdin.

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

- Twenty-one automated tests passed after the VieNeu integration; dependency
  checks reported no broken requirements. The ten-second English replay with
  VieNeu completed five translated segments through direct PipeWire playback,
  producing only JSON logs and configuration.
- Matched-text GPU benchmarks measured 30 warm requests per model. Median first
  audio was 36.71 ms for VieNeu and 40.44 ms for MMS; complete generation was
  401.98 ms and 40.47 ms respectively. These measure different output boundaries
  and do not establish speech quality. See [the TTS benchmark](tts-benchmark.md).
- Sixteen automated tests passed after the direct-playback change. `pip check`
  previously reported no broken requirements
  in the tested environment.
- Real model inference completed on an NVIDIA RTX 4070 SUPER with 12 GB VRAM,
  including both conservative and stable-prefix file replay.
- An earlier ten-second file replay with the queued/WAV implementation and
  `--realtime --play` completed with five segments
  submitted successfully to PipeWire. Its first waveform was ready after
  4.107 seconds of input processing, excluding model loading.
- A new ten-second replay completed with five segments played directly from
  in-memory PCM. Only `events.jsonl` and `configuration.json` were created;
  first waveform availability was 4.175 seconds, excluding model loading.
- Recorded outputs and stage statistics are documented in
  [the experiment report](docs/experiment.md). Short sample runs do not establish
  sustained live performance or physical speaker latency.

### Known limitations

- FastSpeech 2, the proposed alVoCat vocoder, and Nemotron are not integrated.
  MMS-TTS is the current TTS baseline.
- Silence detection uses RMS energy rather than a trained VAD. Forced ASR
  boundaries and independent translation segments can reduce translation quality.
- MMS-TTS generates full segments before playback; VieNeu streams chunks.
  Longer translated speech can
  increase text-queue wait and playback delay; automatic duration adaptation is
  not implemented.
- Voice cloning, speaker separation, video synchronization, and background-music
  preservation are not implemented.
- Long-session performance, microphone capture, and Vietnamese → English have
  not been validated in the recorded experiments. Upstream startup warnings
  may still appear.
