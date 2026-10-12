# Changelog

## [0.0.2] - 2026-10-12

- Added streaming ASR → translation → speech in Python.
- Replaced default MMS-TTS with VieNeu v3 Turbo; MMS remains optional.
- Removed the audio output queue and audio saving; playback uses in-memory PCM.
- Added architecture diagrams and TTS benchmarks: VieNeu starts ~9% sooner; MMS completes synthesis ~10× faster on the test GPU.
- Verified 21 tests and real-time file playback.

Details: [development history](docs/development-history.md) · [TTS benchmark](docs/tts-benchmark.md).
