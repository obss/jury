import inspect
import json
import os
from typing import Optional

import pytest

from jury import Jury, load_metric
from tests.jury import EXPECTED_OUTPUTS

_TEST_METRICS = [
    load_metric("accuracy"),
    load_metric("bertscore", compute_kwargs={"model_type": "albert-base-v1", "device": "cpu"}),
    load_metric("bleu"),
    load_metric("f1"),
    load_metric("meteor"),
    load_metric("precision"),
    load_metric("recall"),
    load_metric("rouge"),
    load_metric("sacrebleu"),
    load_metric("squad"),
]

_STR_TEST_METRIC = "bleu"

_LIST_STR_TEST_METRICS = [
    "accuracy",
    "bertscore",
    "bleu",
    "f1",
    "meteor",
    "precision",
    "recall",
    "rouge",
    "sacrebleu",
    "squad",
]

_LIST_DICT_TEST_METRICS = [
    {"metric_name": "accuracy"},
    {"metric_name": "bertscore", "compute_kwargs": {"model_type": "albert-base-v1"}},
    {"metric_name": "bleu", "resulting_name": "bleu-1", "compute_kwargs": {"max_order": 1}},
    {"metric_name": "bleu", "resulting_name": "bleu-2", "compute_kwargs": {"max_order": 2}},
    {"metric_name": "f1", "resulting_name": "F1"},
    {"metric_name": "meteor", "resulting_name": "METEOR"},
    {"metric_name": "precision"},
    {"metric_name": "recall"},
    {"metric_name": "rouge"},
    {"metric_name": "sacrebleu"},
    {"metric_name": "squad"},
]

_LIST_MIXED_TEST_METRICS = [
    "accuracy",
    "bertscore",
    "bleu",
    {"metric_name": "f1"},
    {"metric_name": "meteor"},
    {"metric_name": "precision"},
    "recall",
    "rouge",
    {"metric_name": "sacrebleu"},
    {"metric_name": "squad"},
]

_DATASETS_METRICS = "wer"

_TEST_METRICS_SEQUENCE_CLASSIFICATION = [
    {"metric_name": "accuracy", "task": "sequence-classification"},
    {"metric_name": "f1", "task": "sequence-classification"},
    {"metric_name": "precision", "task": "sequence-classification"},
    {"metric_name": "recall", "task": "sequence-classification"},
]


@pytest.fixture(scope="package")
def predictions():
    return ["There is a cat on the mat.", "Look! a wonderful day."]


@pytest.fixture(scope="package")
def references():
    return ["The cat is playing on the mat.", "Today is a wonderful day"]


@pytest.fixture
def predictions_sequence_classification():
    return [0, 2, 1, 0, 0, 1]


@pytest.fixture
def references_sequence_classification():
    return [0, 1, 2, 0, 1, 2]


@pytest.fixture
def multiple_predictions_sequence_classification():
    return [[0], [1, 2], [0], [1], [0], [1, 2]]


@pytest.fixture
def multiple_references_sequence_classification():
    return [[0, 2], [1, 0], [0, 1], [0], [0], [1, 2]]


@pytest.fixture(scope="function")
def inconsistent_predictions():
    return ["There is a cat on the mat."]


@pytest.fixture(scope="function")
def single_prediction_array():
    return [["the cat is on the mat"], ["Look! a wonderful day."]]


@pytest.fixture(scope="package")
def multiple_predictions():
    return [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]


@pytest.fixture(scope="package")
def multiple_references():
    return [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]


@pytest.fixture(scope="module")
def jury():
    return Jury(metrics=_TEST_METRICS)


@pytest.fixture(scope="function")
def jury_concurrent():
    return Jury(metrics=_TEST_METRICS, run_concurrent=True)


@pytest.fixture(scope="function")
def jury_str():
    return Jury(metrics=_STR_TEST_METRIC)


@pytest.fixture(scope="function")
def jury_list_str():
    return Jury(metrics=_LIST_STR_TEST_METRICS)


@pytest.fixture(scope="function")
def jury_list_dict():
    return Jury(metrics=_LIST_DICT_TEST_METRICS)


@pytest.fixture(scope="function")
def jury_list_mixed():
    return Jury(metrics=_LIST_MIXED_TEST_METRICS)


@pytest.fixture(scope="function")
def jury_datasets():
    return Jury(metrics=_DATASETS_METRICS)


@pytest.fixture(scope="function")
def jury_sequence_classification():
    return Jury(metrics=_TEST_METRICS_SEQUENCE_CLASSIFICATION)


def get_expected_output(prefix: Optional[str] = None):
    def json_load(path: str):
        with open(path, "r") as jf:
            content = json.load(jf)
        return content

    def wrapper(fn, *args, **kwargs):
        module_name = os.path.basename(inspect.getfile(fn)).replace(".py", "")
        path = os.path.join(EXPECTED_OUTPUTS, prefix, f"{module_name}.json")
        test_name = fn.__name__.replace("output_", "")
        fn.output = json_load(path)[test_name]
        return fn

    if prefix is None:
        prefix = ""
    return wrapper
