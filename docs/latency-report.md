# Latency report (2026-10-13, RTX 5060 Ti, Windows 10)

"Latency" below = from the moment a sentence **stops being spoken** to the moment the first Vietnamese sample is
handed to the speaker. The older `commit_to_audio_ready_ms` metric omits the time spent waiting for the
transcript to stabilise and under-reports what a listener feels.

## 1. Real app runs (your sessions, `output/app-*`)
| run | wall | segments | dropped | speech rate | translation queue p50 / p95 | commit→audio p50 / p95 | TTS busy |
|---|---|---|---|---|---|---|---|
| 233908 | 200 s | 77 | 27 | 141 wpm | 5.5 s / 7.6 s | 6.8 s / 8.8 s | 64 % |
| 234350 | 46 s | 25 | 0 | 159 wpm | 1.8 s / 4.6 s | 2.9 s / 5.4 s | 71 % |
| 234501 | 119 s | 31 | 2 | 76 wpm | 2.1 s / 7.0 s | 2.8 s / 7.8 s | 35 % |

Model stages are not the problem: translation p50 94–172 ms, TTS first chunk p50 110–125 ms, ASR call p50 330–410 ms.
Echo cancellation was locked 100 % of the time with 0 audio glitches.

## 2. Controlled probe (`scripts/latency_probe.py`, 14 known sentences, real models, simulated real-time playback)
| speech style | speech end → speaker p50 | p95 | max | wait for commit p50 | queue wait p50 | translate p50 | TTS 1st chunk p50 |
|---|---|---|---|---|---|---|---|
| sentences 2.5 s apart (pauses) | **1.0 s** | 1.8 s | 2.5 s | 0.8 s | 0.0 s | 0.14 s | 0.08 s |
| sentences 0.35 s apart (continuous) | **3.5 s** | 7.2 s | 13.4 s | 1.4 s | 1.5 s (grows to 6.3 s) | 0.15 s | 0.08 s |

Vietnamese audio is ~0.9× as long as the English speech, and 14 sentences were split into 20–23 segments at commas.

## 3. What this shows
1. **Floor with pauses ≈ 1 s**: 0.6 s silence rule + decode + 0.14 s translate + 0.08 s TTS.
2. **Continuous speech builds a backlog.** One worker translates → synthesizes → *plays* each segment before starting
   the next, so every segment adds ~0.25 s of dead time (translate + first chunk) on top of audio that already
   fills ~90 % of real time. Utilisation approaches 100 %, the queue grows ~0.4 s per sentence, then sentences older
   than 8 s are dropped (27 of 104 in the longest run).
3. **Comma splitting** makes one sentence 2–3 segments; the listener waits for the last piece, and short pieces
   ("Yesterday,", "This video is.") translate poorly.
4. **Wait for commit rises to 1.4-1.8 s** without silence: the window only commits on punctuation
   agreement across two ASR decodes, or at 24 words.

Caveats: the probe uses synthetic SAPI speech and a simulated speaker (no devices). (The "10 s" commit wait and
"max 13.4 s" first reported for continuous speech came from a matching bug, fixed in section 4.) Reproduce: `.venv\Scripts\python scripts\latency_probe.py`
(sentence wavs in `.cache/latency/`).

## 4. After the change: independent stages + adaptive speed (2026-10-14)

Same probe, same 14 sentences. **Caveat:** these runs happened while another game/video was using the GPU (48 % busy),
so every GPU stage was ~2.5x slower than in section 2 (translate p50 0.36 s instead of 0.14 s). Compare rows *within*
this table, not with section 2. The scoring also had a bug that attached short fragments ("This video is.") to the
wrong sentence and inflated the old "max 13 s"; it is fixed, and section 2's numbers were re-scored: old serial pipeline,
sentences 0.35 s apart: p50 2.8 s, p95 6.6 s, max 7.2 s, translation-queue wait up to 6.6 s.

| speech end -> first Vietnamese audio | p50 | p95 | max | wait for commit | speaker wait | speed p50 / max |
|---|---|---|---|---|---|---|
| continuous, 1.0x (new stages only) | 3.3 s | 5.8 s | 5.9 s | 1.8 s | 0.21 s | 1.00 / 1.00 |
| continuous, default 1.1x -> 1.4x, comma cuts | **2.7 s** | **3.9 s** | **4.1 s** | 1.7 s | 0.00 s | 1.10 / 1.20 |
| continuous, default speed + semantic cuts | **2.4 s** | 4.2 s | 4.8 s | 1.6 s | 0.00 s | 1.10 / 1.20 |
| pauses, 1.0x | 1.5 s | 2.0 s | 2.8 s | 0.9 s | 0.00 s | 1.00 / 1.00 |
| pauses, default 1.1x -> 1.4x | 1.4 s | 1.9 s | 2.9 s | 0.8 s | 0.00 s | 1.10 / 1.15 |
| pauses, default speed + semantic cuts | **1.2 s** | **1.4 s** | 1.4 s | 0.8 s | 0.00 s | 1.10 / 1.10 |

- Translation-queue wait is 0 in every scenario (was up to 6.6 s): translation no longer waits for the speaker.
- With continuous speech the backlog no longer grows; speed settles around x1.1-1.2 and the tail (p95) falls from
  6.6 s (old) / 5.8 s (1.0x) to 3.9 s. Vietnamese audio for the same speech: 43.2 s at 1.0x vs 39.1 s with adaptive speed.
- What remains is mostly **waiting for the ASR to commit** (0.8 s with pauses, ~1.7 s in continuous speech): that is
  the next lever (smaller ASR chunk than 1 s, VAD) rather than TTS or the speaker.
- Semantic cuts: 17 pieces instead of 23, and translations stay whole; p50 is lower, the tail is a bit higher.

Real-device check of the app (2026-10-14): base speed 1.1x, commit -> speaker p50 0.27 s, echo locked, 0 glitches.
