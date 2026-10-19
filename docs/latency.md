# What do the timing numbers mean?

The current pipeline uses Qwen ASR, NLLB translation, and **VieNeu TTS**.
MMS results belong to the earlier comparison.

## Translation queue: waiting for a turn

`translation_queue_ms` measures how long a finished text segment waits before
NLLB starts translating it. It is not translation computation time.

The worker handles one segment at a time: translate it, generate speech, and
finish playing it. Meanwhile, ASR keeps listening and can finish more text.
Those new segments wait in the **text queue**.

For example, if the previous segment still has two seconds left to play,
the next segment may wait about two seconds before translation starts.
Removing the audio output queue did not remove this text queue.

In the recorded VieNeu run, the middle queue-wait value was about **1.07 seconds**.
NLLB translation itself took about **73 milliseconds**. Most of that waiting
was outside the translation model.

## Commit to audio ready: from finished text to the last audio chunk

Here, **commit** means ASR has finished a text segment and sent it for translation.
It has nothing to do with a Git commit. **Waveform** simply means audio samples.

`commit_to_audio_ready_ms` measures:

```text
Text finished → wait for a turn → translate → receive the last VieNeu audio chunk
```

Example numbers, just to explain the calculation:

| Part | Time |
|---|---:|
| Wait for a turn | 2.00 seconds |
| Translate the text | 0.07 seconds |
| Receive all audio chunks | 0.60 seconds |
| Commit to audio ready | 2.67 seconds |

VieNeu does not need all chunks before playback starts. If its first chunk
arrives after 0.04 seconds, audio can be submitted to the player about
**2.11 seconds** after the text was finished in this example.
Actual speaker output also depends on the player's own buffers.

Receiving the last chunk does not mean the speaker has finished playing it.
Also, this metric starts after ASR finishes the text: earlier listening and
ASR waiting are not included.

## Which number should I look at?

| Log field | Plain meaning |
|---|---|
| `translation_queue_ms` | Waiting before translation starts |
| `translation_ms` | Translating the text with NLLB |
| `tts_first_chunk_ms` | Waiting for VieNeu's first audio chunk, after translation |
| `tts_stream_elapsed_ms` | From translation completion to the last audio chunk; can include waiting for the player |
| `commit_to_audio_ready_ms` | Waiting + translation + time until the last audio chunk |
| `playback_ms` | Time spent in the player call; generation can overlap this |
| `first_audio_ready_seconds_from_input_start` | From starting the input to having the first translated audio chunk |

`ms` means milliseconds: **1,000 ms = 1 second**.
`p50` is the middle value; `p95` is a value that about 95% of measurements do not exceed.
Do not add streaming time and playback time: they overlap.
Model loading is excluded from these processing timings.

The main delay is often waiting for stable ASR text or for previous speech to
finish. A faster TTS model alone does not remove those waits. Future options
include shorter speech, adjusting speaking speed, or overlapping preparation
of the next segment. These are not implemented yet.
