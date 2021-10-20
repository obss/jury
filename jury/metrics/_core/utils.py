import importlib.util
import os
import re
import string
import warnings
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import requests

PACKAGE_CORE = Path(os.path.abspath(os.path.dirname(__file__)))
METRICS_ROOT = PACKAGE_CORE.parent
PACKAGE_SOURCE = METRICS_ROOT.parent
PROJECT_ROOT = PACKAGE_SOURCE.parent


class PackagePlaceholder:
    def __init__(self, version: str):
        self.__version__ = version


class TaskNotAvailable(KeyError):
    def __init__(self, metric_name: str, task: str):
        message = f"Task '{task}' is not available for metric '{metric_name}'."
        self.message = message
        super(TaskNotAvailable, self).__init__(message)


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
    def remove_punctuations_and_ws(s: str) -> str:
        pattern = r"[%s]" % re.escape(string.punctuation)
        s = re.sub(pattern, " ", s)
        return " ".join(s.split())

    return remove_punctuations_and_ws(text).lower()


def is_reduce_fn(fun: Callable) -> bool:
    result = np.array(fun([1, 2]))
    return result.size == 1


def list_metrics():
    _internal_metrics_path = METRICS_ROOT
    metric_modules = list(_internal_metrics_path.glob("[!_]*"))
    return [module_name.name.replace(".py", "") for module_name in metric_modules]
