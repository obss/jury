import re
import string


def remove_punctuations(text: str) -> str:
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    text = regex.sub(" ", text)
    return " ".join(text.split())
