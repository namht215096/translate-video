# MMS-TTS and VieNeu-TTS speed comparison

Measured on 2026-10-12 using an NVIDIA GeForce RTX 4070 SUPER (12 GB),
Python 3.12, PyTorch 2.9.1, Transformers 4.57.6, and VieNeu SDK 3.8.3.
The current pipeline uses VieNeu. MMS is the former model, retained only for
reproducing this comparison. For a simple explanation of timing names, see
[the timing guide](latency.md).

## Method

`scripts/benchmark_tts.py` runs six identical Vietnamese texts, from short
greetings to a 42-word paragraph, five times each: **30 measured requests per
model**. Both models use two CPU threads and the same shuffled text order and
per-request seed. Each backend runs separately, without ASR, translation,
playback, or saved audio. CUDA is synchronized at request boundaries.

The first request is recorded separately and is followed by three warmup
requests. Warm measurements run consecutively without intentional idle gaps.
They do not predict the first request after a long GPU idle period.

- MMS: `facebook/mms-tts-vie`, float32 on CUDA, 16 kHz whole-waveform output.
- VieNeu: `pnnbao-ump/VieNeu-TTS-v3-Turbo`, SDK default `update` subfolder,
  CUDA backend, `infer_stream`, one scheduler slot and one batch slot,
  preset `Hải Đăng`, 48 kHz, watermark disabled.

The same seed does not produce matching voices or durations across different
architectures. The JSON artifacts record checkpoint revisions and package
versions. Audio is consumed immediately and discarded; nothing is written as WAV.

## Warm results

| Metric | MMS-TTS median / p95 | VieNeu median / p95 |
|---|---:|---:|
| First available audio | 40.44 / 58.07 ms | 36.71 / 47.46 ms |
| Complete generation | 40.47 / 58.11 ms | 401.98 / 1569.38 ms |
| Generated audio duration | 4.26 / 14.78 s | 2.40 / 9.68 s |
| Real-time factor (generation time / audio duration) | 0.0088 / 0.0366 | 0.1678 / 0.2034 |

MMS returns the entire waveform at its first-audio boundary; VieNeu returns a
small initial chunk. VieNeu reduces median first-audio latency by approximately
**9% (3.73 ms)** in this setup. MMS completes synthesis about **10 times faster**
by aggregate median time. Both generate faster than playback (RTF below 1).
VieNeu's streaming support does not imply faster total synthesis.

The separately recorded first requests took 517.66 ms (MMS) and 660.59 ms
(VieNeu) to first audio. VieNeu's initial request includes lazy scheduler and
CUDA graph initialization; model downloads/loading are excluded. These are
single observations, not a cold-start distribution.

| Text | MMS first / complete, median | VieNeu first / complete, median |
|---|---:|---:|
| Xin chào. | 41.1 / 41.2 ms | 36.7 / 161.2 ms |
| Ồ, đúng rồi. | 40.4 / 40.4 ms | 37.8 / 180.2 ms |
| Hôm nay trời đẹp… | 39.5 / 39.5 ms | 35.4 / 358.6 ms |
| Anh ta thậm chí… | 40.5 / 40.5 ms | 36.5 / 612.2 ms |
| nhưng và âm nhạc solo… | 40.2 / 40.3 ms | 37.2 / 418.1 ms |
| Long paragraph | 57.5 / 57.6 ms | 37.1 / 1544.8 ms |

Raw requests, complete texts, and per-text results:
[MMS JSON](results/tts-mms.json), [VieNeu JSON](results/tts-vieneu.json).

## Full pipeline validation

The 10-second English sample was replayed in real time with Qwen ASR, NLLB,
VieNeu, and direct PipeWire playback. The run exited successfully and completed
five segments. The output directory contained exactly `configuration.json` and
`events.jsonl`; no audio files were created.

- First chunk available: 4.029 s after input processing began, excluding model loading.
- TTS first chunk: median 43.41 ms, p95 322.33 ms.
- Input plus drain/playback: 13.89 s; generated speech: 8.24 s.

See [the pipeline summary](results/vieneu-realtime.json). The earlier MMS run
took 18.80 s and generated 13.52 s of speech; see
[the historical MMS summary](results/direct-playback.json). That wall-time
difference is influenced by generated speech duration, voice, random generation,
and playback. It is not an isolated TTS computation speed comparison.

## Interpretation and limits

VieNeu supports incremental output and selectable Vietnamese voices. In this
comparison, the former MMS model needed less time to generate a complete sentence.
The claim that VieNeu is simply faster than MMS is not supported by this test.

These are small, single-machine measurements, without confidence intervals or
speech quality scoring. VieNeu generated shorter speech for the same texts;
this test does not establish whether the difference comes only from speaking
rate or includes pronunciation/content errors. Listen to both before choosing
for quality. ASR stabilization, translation, text backlog, and physical speaker
latency are excluded from the isolated benchmark. Streaming pipeline elapsed
time can include player backpressure and overlaps playback time.

The upstream benchmark uses different hardware and scheduler settings:
[official streaming measurements](https://github.com/pnnbao97/VieNeu-TTS/blob/main/docs/streaming.md).
Those numbers are not substituted for the local measurements above.

## Reproduction

Install the project with `.[vieneu]`; retain vLLM's PyTorch 2.9.1 rather than
installing VieNeu's CUDA extra, which pins PyTorch 2.8.0. This configuration
passed dependency checks and real GPU inference in this project.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_tts.py \
  --backend mms --output output/bench-mms.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_tts.py \
  --backend vieneu --output output/bench-vieneu.json
```

Run sequentially while other GPU inference is stopped. Use new output filenames;
remove offline mode on the first run to download uncached model artifacts.
