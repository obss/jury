import importlib.util
import os
import re
import string
import warnings
from typing import Callable, Sequence, Union

import numpy as np
import requests


class PackagePlaceholder:
    def __init__(self, version: str):
        self.__version__ = version


def requirement_message(metric_name: str, package_name: str) -> str:
    return (
        f"In order to use metric '{metric_name}', '{package_name}' is required. "
        f"You can install the package by `pip install {package_name}`."
    )


def download(source: str, destination: str, overwrite: bool = False, warn: bool = False) -> None:
    if os.path.exists(destination) and not overwrite:
        if warn:
            warnings.warn(
                f"Path {destination} already exists, not overwriting. To overwrite, speficy " f"'overwrite' parameter."
            )
        return
    r = requests.get(source, allow_redirects=True)

    with open(destination, "wb") as out_file:
        out_file.write(r.content)


def import_module(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
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


def is_reduce_fn(fun: Callable) -> bool:
    result = np.array(fun([1, 2]))
    return result.size == 1
