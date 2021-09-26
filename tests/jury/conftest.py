import functools
import inspect
import json
import os
from typing import Callable, Optional

import pytest

from jury import Jury
from jury.metrics import load_metric
from tests.jury import EXPECTED_OUTPUTS

_TEST_METRICS = [
    load_metric("accuracy"),
    load_metric("bertscore", params={"model_type": "albert-base-v1", "device": "cpu"}),
    load_metric("bleu"),
    load_metric("f1"),
    load_metric("meteor"),
    load_metric("precision"),
    load_metric("recall"),
    load_metric("rouge"),
    load_metric("sacrebleu"),
    load_metric("squad"),
]

_CONCURRENT_TEST_METRICS = [
    load_metric("bleu"),
    load_metric("meteor"),
    load_metric("rouge"),
    load_metric("sacrebleu"),
    load_metric("bertscore", params={"model_type": "albert-base-v1", "device": "cpu"}),  # Memory allocation
    load_metric("squad"),
]

_STR_TEST_METRICS = ["bleu", "meteor", "rouge", "sacrebleu", "squad"]


@pytest.fixture(scope="session")
def predictions():
    return ["There is a cat on the mat.", "Look! a wonderful day."]


@pytest.fixture(scope="session")
def references():
    return ["The cat is playing on the mat.", "Today is a wonderful day"]


@pytest.fixture(scope="session")
def single_prediction_array():
    return [["the cat is on the mat"], ["Look! a wonderful day."]]


@pytest.fixture(scope="session")
def multiple_predictions():
    return [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]


@pytest.fixture(scope="session")
def multiple_references():
    return [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]


@pytest.fixture(scope="package")
def jury():
    return Jury(metrics=_TEST_METRICS)


@pytest.fixture(scope="package")
def jury_str():
    return Jury(metrics=_STR_TEST_METRICS)


@pytest.fixture(scope="package")
def jury_concurrent():
    return Jury(metrics=_CONCURRENT_TEST_METRICS, run_concurrent=True)


def json_load(path: str):
    with open(path, "r") as jf:
        content = json.load(jf)
    return content


def get_expected_output(prefix: Optional[str] = None):
    def wrapper(fn, *args, **kwargs):
        module_name = os.path.basename(inspect.getfile(fn)).replace(".py", "")
        path = os.path.join(EXPECTED_OUTPUTS, prefix, f"{module_name}.json")
        test_name = fn.__name__.replace("output_", "")
        fn.output = json_load(path)[test_name]
        return fn

    if prefix is None:
        prefix = ""
    return wrapper
