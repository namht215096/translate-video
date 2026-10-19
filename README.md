# translate-video

Translate English speech into spoken Vietnamese with Python.
The current project uses **VieNeu-TTS-v3-Turbo** for speech output.

```text
Audio → Qwen ASR → English text → NLLB → Vietnamese text → VieNeu → Speaker
```

ASR listens while a second worker translates and reads each finished text segment.
VieNeu sends small audio chunks to the speaker as they are generated.
The project saves text and timing logs, but does not save audio.

![Architecture](docs/figures/architecture.png)

## Use the web interface

```bash
.venv/bin/python -m pip install -e '.[web,vieneu]'
./start-backend.sh
```

The backend runs in the background. Use `./stop-backend.sh` to stop it.
Logs are in `output/backend/backend.log`.

Open **http://127.0.0.1:8000**, drag an MP3 into the page, then click
**Phát & dịch**. You can pause, stop, adjust each voice's volume, and read the
English text and Vietnamese translation. The page shows **P95 latency**.

The app processes a short part first, then starts both voices when Vietnamese
audio is ready. It reads up to five seconds ahead in the file to reduce delay.
This works for recorded files, not live microphone audio.

The models run on this computer's GPU and load when the server starts. Wait
for **Model đã sẵn sàng** before playing. Speech is translated as the file plays, with a delay;
it is not synchronized dubbing. Audio stays in memory. See
[the short web guide](docs/web.md).

## Run the 10-second example

On the configured project machine:

```bash
CC="$PWD/scripts/zig-cc" HF_HUB_OFFLINE=1 .venv/bin/translate-video \
  --input examples/english_10s.wav --realtime --play \
  --output "output/live-$(date +%Y%m%d-%H%M%S)"
```

- `CC`: use the project's Zig compiler wrapper.
- `HF_HUB_OFFLINE=1`: use downloaded models. Remove this on the first run.
- `--input`: the source audio file.
- `--realtime`: read the file at its normal speed.
- `--play`: hear the Vietnamese speech.
- `--output`: a new folder for JSON logs, not audio files.

Wait for the models to load. The command plays the translated Vietnamese speech.
To hear the original English separately, run `pw-play examples/english_10s.wav`.

To choose another VieNeu voice, add `--tts-voice 'Mai Anh'`.
To use the microphone instead of a file, replace `--input ... --realtime` with
`--mic`. Press Ctrl+C to finish. Microphone capture needs PortAudio and has not
been tested on the project machine.

## First-time setup

Use Linux, Python 3.12, and an NVIDIA CUDA GPU. The project has been tested on
an RTX 4070 SUPER with 12 GB VRAM.

```bash
python3.12 -m venv .venv
mkdir -p .cache/pip-tmp
TMPDIR="$PWD/.cache/pip-tmp" .venv/bin/python -m pip install -e '.[dev,compiler,vieneu]'
export CC="$PWD/scripts/zig-cc"
```

The temporary folder keeps large downloads off the RAM-backed `/tmp`.
If GCC or Clang is installed, the optional Zig compiler is not required.
Keep the project's PyTorch 2.9.1: installing `vieneu[cuda]` would request
PyTorch 2.8.0 and conflict with vLLM.

Download the public English example if no input file is available:

```bash
mkdir -p examples
curl -fL https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav \
  -o examples/asr_en.wav
```

Then run the command above with `--input examples/asr_en.wav` and without
`HF_HUB_OFFLINE=1`. This source file is about 15 seconds; the prepared
`english_10s.wav` is a shorter excerpt.

Playback uses PipeWire (`pw-play`), ALSA (`aplay`), or PortAudio.
On Ubuntu, microphone capture may need `libportaudio2`.

## Understand the timing numbers

- **Translation queue:** how long a text segment waits for its turn. The worker
  may still be reading the previous segment aloud.
- **Translation time:** how long NLLB takes to translate that segment.
- **First audio:** how long VieNeu takes to return its first audio chunk.
- **Commit to audio ready:** from ASR finishing a text segment to receiving its
  last audio chunk. This includes waiting in the text queue.

For streaming, the first chunk can already be playing before the last chunk
arrives. A long queue wait does not mean the translation model itself is slow.

See [timing explained with an example](docs/latency.md).
To summarize one run:

```bash
.venv/bin/python -m translate_video.report output/YOUR_RUN/events.jsonl
```

`events.jsonl` stores text and timings. `configuration.json` stores the run settings.

## What has been tested

- 30 automated tests passed: `.venv/bin/python -m pytest`.
- Real-time replay of the 10-second English sample completed with VieNeu playback.
- The [TTS benchmark](docs/tts-benchmark.md) compares VieNeu with the former MMS model.
  MMS is retained in code for reproducing that comparison; normal runs use VieNeu.

This prototype does not yet synchronize video, clone speakers, or adjust speech
speed to prevent delays. It waits for stable ASR text before translating, so
spoken output is delayed. Long microphone sessions have not been validated.

## More information

- [Timing guide](docs/latency.md)
- [Benchmark results](docs/tts-benchmark.md)
- [Earlier experiments](docs/experiment.md)
- [Development history](docs/development-history.md)
- [Changelog](CHANGELOG.md)

Models: [Qwen ASR](https://github.com/QwenLM/Qwen3-ASR),
[NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M),
[VieNeu](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo).
NLLB uses a non-commercial license; this project is for non-commercial evaluation.
