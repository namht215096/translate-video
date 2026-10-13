"""Commit stable transcript prefixes without relying exclusively on commas."""

import re


class RevisionError(RuntimeError):
    """ASR changed text that was already committed."""


MODES = ("punctuation", "semantic")
SENTENCE_END = re.compile(r"[.;!?。！？][\"'”’)]*$")
CLAUSE_END = re.compile(r"[,:，][\"'”’)]*$")
ANY_PUNCTUATION = re.compile(r"[,.;:!?。！？，][\"'”’)]*$")
# A long run of words without punctuation is cut before the last of these, which usually opens a new clause.
CLAUSE_OPENERS = frozenset({
    "and", "but", "because", "so", "which", "while", "when", "although", "though", "if", "since", "then",
    "where", "whereas", "however", "or", "until", "unless", "after", "before",
})


class Segmenter:
    """State belongs to one ASR window, not the entire recording.

    Stability is agreement between consecutive *decode results*, not repeated
    reads of the same result. Once committed, speech cannot be revised.

    mode="punctuation": cut at every , . ; : ! ? (lowest delay, but clauses are translated alone).
    mode="semantic": cut at sentence ends, and at a comma/colon only once the clause has at least
    `min_words` words; a long run without punctuation is cut before a clause opener (and, but, because...)
    instead of at an arbitrary word. Fewer, more complete pieces; waits a little longer for them.
    """

    def __init__(self, max_words: int = 24, mode: str = "punctuation", min_words: int = 8,
                 soft_words: int = 16):
        if max_words < 1:
            raise ValueError("max_words must be positive")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.max_words = max_words
        self.mode = mode
        self.min_words = min_words
        self.soft_words = min(soft_words, max_words)
        self.previous: list[str] = []
        self.committed: list[str] = []

    def update(self, text: str, final: bool = False) -> list[str]:
        words = text.split()
        if words[:len(self.committed)] != self.committed:
            raise RevisionError(
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
            end = self.find_end(words, start, stable, final)
            if end is None:
                break
            end = min(end, start + self.max_words)
            emitted.append(" ".join(words[start:end]))
            self.committed = words[:end]
            start = end
        return emitted

    def find_end(self, words, start, stable, final):
        """Exclusive end index of the next piece, or None if more stable words are needed."""
        if self.mode == "punctuation":
            end = next((i + 1 for i in range(start, stable) if ANY_PUNCTUATION.search(words[i])), None)
        else:
            end = next(
                (i + 1 for i in range(start, stable)
                 if SENTENCE_END.search(words[i])
                 or (CLAUSE_END.search(words[i]) and i + 1 - start >= self.min_words)),
                None,
            )
        if end is not None:
            return end
        if final:
            return min(stable, start + self.max_words)
        if self.mode == "semantic" and stable - start >= self.soft_words:
            opener = next((i for i in range(stable - 3, start + self.min_words - 1, -1)
                           if words[i].lower().strip(",.;:") in CLAUSE_OPENERS), None)
            if opener is not None:
                return opener
        if stable - start >= self.max_words:
            return start + self.max_words
        return None
