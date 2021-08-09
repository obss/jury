import numpy as np
import pytest

from jury import Jury
from jury.metrics import load_metric
from tests.jury import _DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES

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
    load_metric("sacrebleu"),
    # load_metric("bertscore", params={"model_type": "albert-base-v1", "device": "cpu"}),  # Memory allocation
    load_metric("squad"),
]

_STR_TEST_METRICS = ["bleu", "meteor", "rouge", "sacrebleu", "squad"]


def test_evaluate_basic():
    jury = Jury(metrics=_TEST_METRICS)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_basic_str_input():
    jury = Jury(metrics=_STR_TEST_METRICS)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_corpus():
    predictions = [["the cat is on the mat"], ["Look! a wonderful day."]]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]

    jury = Jury(metrics=_TEST_METRICS)
    scores = jury.evaluate(predictions, references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_multiple_predictions():
    predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]
    jury = Jury(metrics=_TEST_METRICS)
    scores = jury.evaluate(predictions=predictions, references=references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_concurrent():
    jury = Jury(metrics=_CONCURRENT_TEST_METRICS, run_concurrent=True)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_reduce_fn():
    _reduce_fn = np.mean
    _non_reduce_fn = np.exp
    jury = Jury(metrics=_TEST_METRICS)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES, reduce_fn=_reduce_fn)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])

    with pytest.raises(ValueError):
        jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES, reduce_fn=_non_reduce_fn)
