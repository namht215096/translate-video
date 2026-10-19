"""Cheap script-level language checks (no model): keep our own Vietnamese out of the loop and pick the NLLB source code."""

import re

# Letters carrying Vietnamese-only marks (plain accents such as é also occur in English loanwords).
VIETNAMESE_MARKS = set("ăâđêôơưạảãậẫẩầấắằẳẵặẹẻẽềếểễệịỉĩọỏõồốổỗộờớởỡợụủũừứửữựỵỷỹỳ")

HAN = "㐀-䶿一-鿿豈-﫿"
KANA = "぀-ヿ"
HANGUL = "가-힯"
LETTER = re.compile(r"[^\W\d_]")
HAN_RE = re.compile(f"[{HAN}]")
KANA_RE = re.compile(f"[{KANA}]")
HANGUL_RE = re.compile(f"[{HANGUL}]")


def looks_vietnamese(text: str) -> bool:
    """True when enough letters are Vietnamese-specific (>= 3 and >= 8% of letters)."""
    letters = LETTER.findall(text.lower())
    marked = sum(1 for letter in letters if letter in VIETNAMESE_MARKS)
    return marked >= 3 and marked >= 0.08 * len(letters)


def contains_cjk(text: str) -> bool:
    return bool(HAN_RE.search(text) or KANA_RE.search(text))


def nllb_source_code(text: str, default: str = "eng_Latn") -> str:
    """NLLB source-language code for `text`, judged by script: a Chinese sentence tagged `eng_Latn` is
    returned unchanged or garbled, so the tag has to follow the text, not the session setting."""
    total = len(LETTER.findall(text))
    if not total:
        return default
    han, kana, hangul = (len(pattern.findall(text)) for pattern in (HAN_RE, KANA_RE, HANGUL_RE))
    if kana and (kana + han) / total >= 0.3:
        return "jpn_Jpan"
    if han / total >= 0.3:
        return "zho_Hans"
    if hangul / total >= 0.3:
        return "kor_Hang"
    return default


TRAILING_CLAUSE = re.compile(r"[,;:，；：、\s]+$")


def prepare_for_translation(text: str) -> str:
    """Drop a trailing comma/enumeration mark. NLLB treats "…基础，" as an unfinished sentence and invents a
    continuation ("… và họ có thể làm cho việc này trở nên tốt hơn"); sentence-final marks (. ! ? 。) are kept."""
    return TRAILING_CLAUSE.sub("", text) or text


_ASR_TAG = "<asr_text>"
_PROMPT_ECHO = re.compile(r"^\W*system\s+user\s+assistant\s*(?:language\s+\w+)?\s*", re.IGNORECASE)


def clean_asr_text(text: str) -> str:
    """Remove artefacts Qwen3-ASR sometimes emits when the audio is noisy or the forced language is wrong:
    its own prompt ("system user assistant language English<asr_text>…"), the <asr_text> marker, and the
    U+FFFD of a UTF-8 character cut in half (it is completed by the next decode)."""
    if _ASR_TAG in text:
        text = text.rsplit(_ASR_TAG, 1)[1]
    text = text.replace("\ufffd", "").replace(_ASR_TAG, "")
    return _PROMPT_ECHO.sub("", text).strip()
