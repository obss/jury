import platform

import numpy as np
import pytest

from jury import Jury
from jury.metrics.bertscore import BERTScore
from jury.metrics.bleu import Bleu
from jury.metrics.meteor import Meteor
from jury.metrics.rouge import Rouge
from jury.metrics.sacrebleu import SacreBLEU
from jury.metrics.squad import SQUAD

_TEST_METRICS = [
    # Bleu(),
    # Meteor(),
    # Rouge(),
    # SacreBLEU(),
    # BERTScore(params={"model_type": "albert-base-v1"}),
    SQUAD(),
]
_STR_TEST_METRICS = ["bleu", "meteor", "rouge", "sacrebleu", "bertscore"]

_DEFAULT_PREDICTIONS = ["Peace in the dormitory, peace in the world."]
_DEFAULT_REFERENCES = ["Peace at home, peace in the world."]


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

    if platform.system() != "Windows":
        jury = Jury(metrics=_TEST_METRICS, run_concurrent=True)
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
