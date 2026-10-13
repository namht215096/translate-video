"""Commit stable transcript prefixes without relying exclusively on commas.

Text is split into tokens the same way for every language: runs of non-space characters are one token
(English words), but each Chinese/Japanese character is its own token because those scripts have no
spaces. A CJK token carries the punctuation that follows it (，。！？…) and the opening quote before it,
so "每个字，" is checked for a comma just like "word," is. Length limits are measured in words: a CJK
character counts as half a word, so `max_words=24` means 24 English words or 48 Chinese characters.
"""

import re


class RevisionError(RuntimeError):
    """ASR changed text that was already committed."""


MODES = ("punctuation", "semantic")

CJK = "㐀-䶿一-鿿豈-﫿぀-ヿ"          # Han ideographs + kana
OPENING = "“‘（(「『《〈【"
CLOSING = "\"'”’）)」』》〉】"
SENTENCE_PUNCT = ".!?;。！？；…"
CLAUSE_PUNCT = ",:，："

TOKEN = re.compile(
    rf"(\s*)((?:[{re.escape(OPENING)}]*[{CJK}](?:[{re.escape(SENTENCE_PUNCT + CLAUSE_PUNCT + CLOSING + '、—')}])*)"
    rf"|[^\s{CJK}]+)"
)
CJK_TOKEN = re.compile(rf"[{re.escape(OPENING)}]*[{CJK}]")
SENTENCE_END = re.compile(rf"[{re.escape(SENTENCE_PUNCT)}][{re.escape(CLOSING)}{re.escape(SENTENCE_PUNCT)}]*$")
CLAUSE_END = re.compile(rf"[{re.escape(CLAUSE_PUNCT)}][{re.escape(CLOSING)}]*$")
ANY_PUNCTUATION = re.compile(rf"[{re.escape(SENTENCE_PUNCT + CLAUSE_PUNCT)}][{re.escape(CLOSING + SENTENCE_PUNCT)}]*$")

# A long run of words without punctuation is cut before the last of these, which usually opens a new clause.
CLAUSE_OPENERS = frozenset({
    "and", "but", "because", "so", "which", "while", "when", "although", "though", "if", "since", "then",
    "where", "whereas", "however", "or", "until", "unless", "after", "before",
})
CJK_OPENERS = frozenset({
    "但是", "可是", "然而", "不过", "而且", "并且", "因为", "所以", "如果", "虽然", "然后", "同时",
    "而是", "于是", "因此", "以及", "或者", "还是", "另外", "此外", "不仅", "尽管", "只要", "除非",
    "但是", "不過", "而且", "並且", "因為", "所以", "如果", "雖然", "然後", "而是", "於是", "因此",
})
CJK_SINGLE_OPENERS = frozenset({"而", "但"})     # "而已" (that's all) is excluded in opens_clause


def tokenize(text: str):
    """[(separator, token)]; the separator is "" where the source had no whitespace (CJK text)."""
    return [(space, token) for space, token in TOKEN.findall(text)]


def weight(token: str) -> float:
    return 0.5 if CJK_TOKEN.match(token) else 1.0


class Segmenter:
    """State belongs to one ASR window, not the entire recording.

    Stability is agreement between consecutive *decode results*, not repeated
    reads of the same result. Once committed, speech cannot be revised.

    mode="punctuation": cut at every , . ; : ! ? (and 。，！？；：…; lowest delay, but clauses are translated alone).
    mode="semantic": cut at sentence ends, and at a comma/colon only once the clause has at least
    `min_words` words; a long run without punctuation is cut before a clause opener (and, but, because,
    但是, 因为...) instead of at an arbitrary word. Fewer, more complete pieces; waits a little longer for them.
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
        tokens = tokenize(text)
        words = [token for _, token in tokens]
        separators = [space for space, _ in tokens]
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
            end = min(end, self.advance(words, start, self.max_words))
            emitted.append(self.join(words, separators, start, end))
            self.committed = words[:end]
            start = end
        return emitted

    # ---- helpers ----
    @staticmethod
    def size(words, start, end) -> float:
        return sum(weight(word) for word in words[start:end])

    @staticmethod
    def advance(words, start, amount) -> int:
        """First index after `start` at which `amount` words have been covered."""
        total = 0.0
        for index in range(start, len(words)):
            total += weight(words[index])
            if total >= amount:
                return index + 1
        return len(words)

    @staticmethod
    def join(words, separators, start, end) -> str:
        parts = [words[start]]
        for index in range(start + 1, end):
            parts.append((" " if separators[index] else "") + words[index])
        return "".join(parts)

    @staticmethod
    def opens_clause(words, index) -> bool:
        word = words[index]
        if CJK_TOKEN.match(word):
            strip = CLAUSE_PUNCT + SENTENCE_PUNCT
            pair = "".join(w.strip(strip) for w in words[index:index + 2])
            if pair in CJK_OPENERS:
                return True
            return word in CJK_SINGLE_OPENERS and not pair.startswith("而已")
        return word.lower().strip(",.;:") in CLAUSE_OPENERS

    def find_end(self, words, start, stable, final):
        """Exclusive end index of the next piece, or None if more stable words are needed."""
        if self.mode == "punctuation":
            end = next((i + 1 for i in range(start, stable) if ANY_PUNCTUATION.search(words[i])), None)
        else:
            end = next(
                (i + 1 for i in range(start, stable)
                 if SENTENCE_END.search(words[i])
                 or (CLAUSE_END.search(words[i]) and self.size(words, start, i + 1) >= self.min_words)),
                None,
            )
        if end is not None:
            return end
        if final:
            return min(stable, self.advance(words, start, self.max_words))
        pending = self.size(words, start, stable)
        if self.mode == "semantic" and pending >= self.soft_words:
            opener = next((i for i in range(stable - 1, start, -1)
                           if self.size(words, start, i) >= self.min_words
                           and self.size(words, i, stable) >= 3
                           and self.opens_clause(words, i)), None)
            if opener is not None:
                return opener
        if pending >= self.max_words:
            return self.advance(words, start, self.max_words)
        return None
