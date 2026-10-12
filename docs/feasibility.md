# Feasibility and evaluation plan

## Intended result

The initial target is one English speaker translated into Vietnamese through a
fixed voice. The application should continue recognizing speech while earlier
segments are being translated and played. A useful initial target is roughly
2–4 seconds of perceived delay; this is a target to evaluate, not an established
performance claim.

High throughput and low latency are different. Synthesizing a four-second audio
clip in 100 ms is fast inference, but playing that clip still takes four seconds.
The system remains useful only if inference and playback backlogs stay bounded.

The current implementation keeps one text queue and uses a single worker for
translation, TTS, and blocking playback. It plays each waveform directly from
memory and then processes the next text segment. Removing the audio queue does
not remove playback delay: pending work can accumulate in the text queue. Only
JSON logs and configuration are saved.

## ASR choice

Start with Qwen3-ASR-1.7B because it was the requested first candidate. Its
official Python API offers streaming through vLLM. The current implementation
re-feeds the accumulated audio when decoding a new chunk and can roll back
trailing text tokens. This prototype finalizes and resets an ASR window after a
pause or a twelve-second cap. That bounds work but sacrifices context at forced
boundaries.

[Qwen streaming implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)

Nemotron 3.5 ASR 0.6B should be the next comparison. Its cache-aware
FastConformer-RNNT processes new chunks while reusing encoder state. The official
model card lists 80, 160, 320, 560, and 1120 ms configurations, built-in
punctuation, and Vietnamese among its transcription-ready locales. Chunk size
is not the total speech-to-translated-audio latency.

[Nemotron model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

Nemotron needs a separate adapter and dependency environment; it is not a
command-line option in this implementation. Qwen's pinned Transformers version
should not be upgraded blindly to accommodate another model.

## Segment commitment

Waiting exclusively for a comma can create unbounded delay. Punctuation in
partial recognition can also change. The prototype checks agreement across two
decode results and commits only a stable boundary or a bounded number of stable
words. At silence or a maximum window length, it finalizes remaining text.

This is a heuristic. An incomplete clause can have stable spelling and still
lack the context needed for translation. Once speech is played, it cannot be
revised. Conservative final-only mode trades longer delay for fewer premature
commits. Test both modes before optimizing inference speed.

RMS-based silence detection is intentionally simple. Replace it with a trained
VAD and retain a short pre-roll buffer before testing noisy or distant speech.

## Translation

NLLB-600M provides a compact multilingual baseline. Download counts should not
choose the final model. Evaluate the actual language pair on conversational
speech, names, numbers, negation, and domain vocabulary. This prototype uses
greedy decoding and independent segments, so it prioritizes a simple experiment
over discourse-level consistency.

The checkpoint has a non-commercial license and its model card describes it as
a research model not released for production deployment. A commercial service
needs a suitable licensed alternative or explicit permission.

[NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M)

## TTS and the original FastSpeech 2 proposal

FastSpeech 2 is a model architecture, not a complete multilingual voice. It
requires a trained acoustic checkpoint, a text/phoneme frontend, and a compatible
vocoder. Matching mel-bin count alone is insufficient: sample rate, FFT and hop
length, frequency bounds, log scaling, and normalization must also agree.

The proposed alVoCat vocoder was trained on Catalan speech and accepts 80-bin
mel features at 22.05 kHz. Its suitability for a Vietnamese acoustic checkpoint
has not been demonstrated. No compatible FastSpeech 2 checkpoint was supplied.

[alVoCat model card](https://huggingface.co/projecte-aina/alvocat-vocos-22khz)

The runnable baseline therefore uses `facebook/mms-tts-vie`, a small Vietnamese
VITS model. It generates whole audio segments and uses a fixed voice. It is not
FastSpeech 2, voice cloning, or streaming waveform generation.

[MMS Vietnamese model card](https://huggingface.co/facebook/mms-tts-vie)

The current default is now VieNeu-TTS-v3-Turbo through SDK 3.8.3. It emits
48 kHz waveform chunks and replaces the complete TTS block, including audio
decoding; an external FastSpeech 2 vocoder is not needed. MMS remains available
through `--tts-backend mms`. Local measurements show slightly lower first-audio
latency for VieNeu, but much faster complete synthesis for MMS. See
[the matched-text benchmark](tts-benchmark.md) for data and limitations.

## Evaluation procedure

1. Verify the chain on one short audio file with final-only commits.
2. Compare one-second and two-second Qwen chunks on identical recordings.
3. Compare stable-prefix commits against final-only mode.
4. Test uninterrupted speech, pauses, background noise, names, and code switching.
5. Run at least 10–15 minutes with playback enabled. Measure queue growth as
   well as average speed; a short successful clip does not establish stability.
6. Add Nemotron and compare recognition quality and stage latency on the same
   audio, hardware, and segmentation policy.
7. Compare TTS checkpoints using intelligibility and naturalness, not only speed.

Report model versions, hardware, language direction, chunk/window settings,
warmup, and whether playback was enabled. Use median and p95 timings over many
segments. Separately evaluate ASR word errors, translation adequacy, speech
quality, and lost/duplicated content.

Precise live latency needs a source timestamp for each committed boundary and
an output-device playback timestamp. The present event log provides stage
timings and source audio cursors, but lacks that alignment. Do not label its
commit-to-audio-ready metric as end-to-end latency.

## Scope for subsequent work

Video synchronization requires delaying the source video and managing output
speech duration. Multiple speakers require diarization and a speaker-to-voice
policy. Preserving music requires source separation and audio mixing. These
are separate capabilities beyond the initial speech translation experiment.
