import numpy as np
import pytest

from jury import Jury

TEST_METRICS = ["bleu_1", "meteor", "rouge"]
_DEFAULT_PREDICTIONS = ["Peace in the dormitory, peace in the world."]
_DEFAULT_REFERENCES = ["Peace at home, peace in the world."]


def test_evaluate_basic():
    jury = Jury(metrics=TEST_METRICS)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_corpus():
    predictions = [["the cat is on the mat"], ["Look! a wonderful day."]]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]

    jury = Jury(metrics=TEST_METRICS)
    scores = jury.evaluate(predictions, references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_multiple_items():
    predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]
    jury = Jury(metrics=TEST_METRICS)
    scores = jury.evaluate(predictions=predictions, references=references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_preload():
    jury = Jury(metrics=TEST_METRICS, preload_metrics=True)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_concurrent():
    jury = Jury(metrics=TEST_METRICS, preload_metrics=True, run_concurrent=True)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_reduce_fn():
    _reduce_fn = np.mean
    _non_reduce_fn = np.exp
    jury = Jury(metrics=TEST_METRICS)
    scores = jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES, reduce_fn=_reduce_fn)

    assert all([scores[metric] is not None for metric in TEST_METRICS])

    with pytest.raises(ValueError):
        jury.evaluate(_DEFAULT_PREDICTIONS, _DEFAULT_REFERENCES, reduce_fn=_non_reduce_fn)
