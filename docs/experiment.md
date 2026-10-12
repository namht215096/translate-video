# Local experiment report

The initial measurements below describe the earlier implementation with an audio queue
and WAV output. The current implementation has removed both features: it plays
audio directly from memory and saves only JSON logs and configuration. Retained
result files are historical records, not benchmarks of the current architecture.

The Python pipeline completed a real **English → Vietnamese** file-replay run
on 2026-10-12. All three model stages ran locally: Qwen3-ASR-1.7B through vLLM,
NLLB-600M, and Vietnamese MMS-TTS. The final run used stable-prefix commits and
produced seven translated audio segments.

This establishes that the chain runs on this machine. It does not establish
production translation quality, sustained live performance, or speaker latency.

## Hardware and configuration

- GPU: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB total VRAM.
- System RAM: approximately 7 GiB, with 4 GiB swap.
- Python: 3.12.15 on Linux.
- Core libraries: Qwen-ASR 0.0.6, vLLM 0.14.0, PyTorch 2.9.1,
  Transformers 4.57.6.
- ASR: one stream, one-second decoding chunks, 1024-token context and batching
  budget, eager mode, GPU memory utilization 0.55.
- Segmentation: stable-prefix agreement across consecutive decode results;
  0.6-second silence threshold; 12-second maximum ASR window.
- NLLB: GPU, FP16, greedy decoding, independent segments.
- MMS-TTS: GPU, fixed Vietnamese voice, full-segment generation.
- Input: Qwen's public English example, 15.051 seconds, originally 48 kHz mono;
  downsampled to 16 kHz and replayed at the source clock rate.
- Speaker playback: disabled. Audio was written to WAV files.

The full argument/version manifest is in
[demo-configuration.json](results/demo-configuration.json). The exact installed
library versions are in [requirements-tested.txt](../requirements-tested.txt).

Model revisions downloaded for this experiment:

| Model | Hugging Face revision |
| --- | --- |
| Qwen/Qwen3-ASR-1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| facebook/nllb-200-distilled-600M | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| facebook/mms-tts-vie | `b58928d033932a49aa8e3d6cf11625b25fe928d2` |

Inference was run with `HF_HUB_OFFLINE=1` after downloading those checkpoints.
The application normally resolves the model's current default revision; these
revision identifiers document the weights used here.

## Measured results

The first translated waveform was ready **4.290 seconds after input processing
started**. Model startup was excluded. This is waveform availability for the
first segment, not a measurement from every source word to a physical speaker.

| Metric | Median | p95 |
| --- | ---: | ---: |
| ASR decode-triggering call | 103.82 ms | 256.53 ms |
| Translation queue wait | 0.22 ms | 865.21 ms |
| NLLB inference | 65.96 ms | 224.64 ms |
| MMS-TTS inference | 68.94 ms | 501.69 ms |
| Commit to waveform ready | 232.83 ms | 1129.05 ms |
| Audio queue wait, without playback | 2.43 ms | 3.62 ms |

These are a small sample of seven translated segments. First-call overhead is
included; no separate translation/TTS warmup was performed. Percentiles should
not be treated as reliable service-level estimates.

Input processing and worker draining took **15.269 seconds**. The concatenated
output contains **19.232 seconds** of speech at 16 kHz. It contains finite,
nonzero samples and all seven output segments were written successfully.

The output is about 4.18 seconds longer than the entire input clip. Since this
run did not play audio, its small audio-queue timings do not demonstrate that
live playback can keep up. Duration control remains necessary for continuous
speech and video synchronization.

The recorded statistics are in [demo.json](results/demo.json). Local generated
files are in `output/demo/`, including `translated.wav`, individual segment WAVs,
the full event log, and the configuration manifest. Generated audio is ignored
by Git.

## What the comparison revealed

An earlier conservative run finalized text at an eight-second window limit.
It split “didn't do overly well” across two ASR windows, and the isolated second
fragment translated with the wrong polarity. This is an observed segmentation
failure, not an inference-speed failure.

The final configuration uses a twelve-second limit and earlier stable-prefix
commits. In this sample, it retained that negative phrase within one translation
segment. This does not guarantee that other forced boundaries are safe.
The earlier measurements are retained in
[conservative-8s.json](results/conservative-8s.json).

The final translation still contains awkward phrasing, including a literal
translation of “big” in a context about an artist's popularity. Translation
fluency and discourse consistency need further evaluation. No formal ASR error
score, translation score, listening panel, or voice-quality score was measured.

## Configuration requirements established by the run

An initial 4096-token ASR context did not fit the reserved memory budget.
The tested 1024-token context and batching budget fit while leaving space for
NLLB and MMS-TTS. ASR weight loading alone reported approximately 3.87 GiB;
that is not the total runtime GPU memory footprint.

The machine had no C compiler, so the optional Zig 0.13 compiler and
`scripts/zig-cc` wrapper were used for Triton's generated C code. Also, `/tmp`
was RAM-backed; the initial pip install exhausted its limit. Moving installation
temporary files onto disk resolved that issue. CUDA wheels obtained through a
mirror were checked against SHA-256 digests from the official PyPI metadata.

## Validation and remaining work

- Eight automated tests passed in the installed project environment.
- `pip check` reported no broken requirements.
- Real inference completed in both conservative and stable-prefix modes.
- The reference run did not exercise microphone capture or speaker playback;
  a subsequent playback test is described below.
- Vietnamese → English was documented but not tested.
- A 10–15 minute run, a trained VAD, playback-duration adaptation, and a Nemotron
  comparison remain necessary before choosing a production architecture.
- FastSpeech 2 plus alVoCat was not tested; no compatible Vietnamese acoustic
  checkpoint was supplied. MMS-TTS is explicitly the baseline used here.

## Subsequent speaker playback test

The prepared ten-second English excerpt was subsequently run with
`--realtime --play`. The pipeline completed successfully, writing and submitting
five translated segments to `pw-play` through PipeWire. Playback and ASR ran in
separate workers, so audio output did not block input processing. PortAudio was
not required for this file-based playback test.

Results are in [playback-10s.json](results/playback-10s.json), and the local output
is in `output/live-10s-verified/`. This verifies successful player execution;
physical speaker latency and microphone capture remain unmeasured. The prepared
input is an excerpt ending mid-sentence, so its final translation is incomplete.

## Direct playback after removing audio storage

The later VieNeu integration and matched-text comparison are documented in
[the TTS benchmark](tts-benchmark.md), including a successful real-time playback
run with the same English excerpt. The MMS results below remain historical.

The MMS pipeline was tested again with the ten-second excerpt and
`--realtime --play`, after removing the audio queue and WAV saving. Five segments
were played successfully by passing raw PCM from memory to PipeWire stdin.
The run directory contained exactly `events.jsonl` and `configuration.json`.

First waveform availability was 4.175 seconds after input processing began;
the run completed in 18.798 seconds including blocking playback. The generated
speech duration was 13.520 seconds. These are single-run observations, excluding
model startup, and do not measure physical speaker latency.

Translation, synthesis, and playback now run sequentially in one worker. Median
text-queue wait was 1776.15 ms in this sample: pending work remains in the text
queue while a segment plays. Removing the audio queue does not eliminate delay.
Results are recorded in [direct-playback.json](results/direct-playback.json).
Sixteen automated tests passed, including direct playback order, error
propagation, raw PCM transfer, and checks that no audio files are created.
