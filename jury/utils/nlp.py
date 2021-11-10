import re
import string


def normalize_text(text: str, uncased: bool = True) -> str:
    pattern = r"[%s]" % re.escape(string.punctuation)
    text = re.sub(pattern, " ", text)
    normalized_text = " ".join(text.split())
    if uncased:
        normalized_text = normalized_text.lower()
    return normalized_text
