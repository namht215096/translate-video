"""Cheap script-level check used to keep our own Vietnamese speech out of the translation loop."""

import re

# Letters carrying Vietnamese-only marks (plain accents such as é also occur in English loanwords).
VIETNAMESE_MARKS = set("ăâđêôơưạảãậẫẩầấắằẳẵặẹẻẽềếểễệịỉĩọỏõồốổỗộờớởỡợụủũừứửữựỵỷỹỳ")


def looks_vietnamese(text: str) -> bool:
    """True when enough letters are Vietnamese-specific (>= 3 and >= 8% of letters)."""
    letters = re.findall(r"[^\W\d_]", text.lower())
    marked = sum(1 for letter in letters if letter in VIETNAMESE_MARKS)
    return marked >= 3 and marked >= 0.08 * len(letters)
