"""Device self-test: calibrate the echo lag, play a WAV through SystemAudio, report how much of it remains.

Other audio playing on the PC is fine and expected (it is what the app translates). The echo is
measured as the least-squares gain of our speech inside the capture, before and after subtraction.

  .venv\\Scripts\\python scripts\\loopback_selftest.py speech.wav [--gain 0.3]
"""

import argparse
import sys
import time

import numpy as np
import soundfile as sf

from translate_video.loopback import SystemAudio


def fitted_gain(signal, reference):
    return float(np.dot(signal, reference) / max(np.dot(reference, reference), 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("speech")
    parser.add_argument("--gain", type=float, default=0.3)
    args = parser.parse_args()
    speech, rate = sf.read(args.speech, dtype="float32")
    speech = speech if speech.ndim == 1 else speech.mean(axis=1)
    with SystemAudio(gain=args.gain) as system:
        canceller = system.canceller
        print(f"device={system.device_name} rate={system.rate} channels out/in={system.out_channels}/{system.in_channels}")
        time.sleep(1.0)
        idle = canceller._read(canceller.captured, canceller.cap_frames - system.rate, system.rate, canceller.cap_frames)
        print(f"other audio on the PC (rms): {np.sqrt(np.mean(idle ** 2)):.4f}")
        lag = system.calibrate()
        print("calibration lag ms:", lag)
        if lag is None:
            sys.exit("calibration failed")
        system.player.play(speech, rate)
        time.sleep(0.5)
        n = int(len(speech) / rate * system.rate)
        end = canceller.cap_frames
        capture = canceller._read(canceller.captured, end - n - 24000, n + 24000, canceller.cap_frames)
        reference = canceller._read(canceller.reference, end - n - 24000 - canceller.lag, n + 24000, canceller.out_frames)
    before = fitted_gain(capture, reference)
    after = fitted_gain(capture - reference, reference)
    print(f"echo gain before {before:.4f}, after {after:+.5f}  ->  {20 * np.log10(abs(before) / max(abs(after), 1e-9)):.1f} dB reduction")


if __name__ == "__main__":
    main()
