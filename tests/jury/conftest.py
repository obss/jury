import pytest

from jury import Jury
from jury.metrics import load_metric

_TEST_METRICS = [
    load_metric("bleu"),
    load_metric("meteor"),
    load_metric("rouge"),
    load_metric("sacrebleu"),
    load_metric("bertscore", params={"model_type": "albert-base-v1", "device": "cpu"}),
    load_metric("squad"),
]

_CONCURRENT_TEST_METRICS = [
    load_metric("bleu"),
    load_metric("meteor"),
    load_metric("rouge"),
    # load_metric("sacrebleu"),
    # load_metric("bertscore", params={"model_type": "albert-base-v1", "device": "cpu"}),  # Memory allocation
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
