import importlib.util
import os
import re
import string
import warnings
from typing import Callable, Union, Sequence, Optional

import requests


def download(source: str, destination: str) -> None:
    r = requests.get(source, allow_redirects=True)

    with open(destination, "wb") as out_file:
        out_file.write(r.content)


def download_and_import_module(source: str, destination: str, module_name: Optional[str] = "", overwrite: bool = False,
                               warn: bool = False):
    if os.path.exists(destination) and not overwrite:
        if warn:
            warnings.warn(f"Path {destination} already exists, not overwriting. To overwrite, speficy "
                          f"'overwrite' parameter.")
    else:
        download(source, destination)
    spec = importlib.util.spec_from_file_location(module_name, destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_token_lengths(sequences: Sequence[Sequence[str]], reduce_fn: Callable = None) -> Union[int, Sequence[int]]:
    token_lengths = [len(item) for item in sequences]
    if reduce_fn is not None:
        return int(reduce_fn(token_lengths))
    return token_lengths


def normalize_text(text: str) -> str:
    def remove_punctuations_and_ws(text: str) -> str:
        pattern = r"[%s]" % re.escape(string.punctuation)
        text = re.sub(pattern, " ", text)
        return " ".join(text.split())

    return remove_punctuations_and_ws(text).lower()
