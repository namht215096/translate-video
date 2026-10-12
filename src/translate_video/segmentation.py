"""Commit stable transcript prefixes without relying exclusively on commas."""

import re


class Segmenter:
    """State belongs to one ASR window, not the entire recording.

    Stability is agreement between consecutive *decode results*, not repeated
    reads of the same result. Once committed, speech cannot be revised.
    """

    def __init__(self, max_words: int = 24):
        if max_words < 1:
            raise ValueError("max_words must be positive")
        self.max_words = max_words
        self.previous: list[str] = []
        self.committed: list[str] = []

    def update(self, text: str, final: bool = False) -> list[str]:
        words = text.split()
        if words[:len(self.committed)] != self.committed:
            raise RuntimeError(
                "ASR revised already committed speech; stop rather than replay or skip words. "
                "Increase chunk duration or use --final-only."
            )
        stable = len(words) if final else 0
        if not final:
            for old, new in zip(self.previous, words):
                if old != new:
                    break
                stable += 1
        self.previous = words
        emitted = []
        start = len(self.committed)
        while start < stable:
            end = next(
                (i + 1 for i in range(start, stable)
                 if re.search(r"[,.;:!?。！？，][\"'”’)]*$", words[i])),
                None,
            )
            if end is None:
                if final or stable - start >= self.max_words:
                    end = min(stable, start + self.max_words)
                else:
                    break
            else:
                end = min(end, start + self.max_words)
            emitted.append(" ".join(words[start:end]))
            self.committed = words[:end]
            start = end
        return emitted
