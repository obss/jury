import numpy as np
import pytest


def test_evaluate_basic(predictions, references, jury):
    scores = jury.evaluate(predictions, references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_basic_str_input(predictions, references, jury_str):
    scores = jury_str.evaluate(predictions, references)

    assert all([scores[metric.resulting_name] is not None for metric in jury_str.metrics])


def test_evaluate_corpus(single_prediction_array, multiple_references, jury):
    scores = jury.evaluate(single_prediction_array, multiple_references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_multiple_predictions(multiple_predictions, multiple_references, jury):
    scores = jury.evaluate(predictions=multiple_predictions, references=multiple_references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_concurrent(predictions, references, jury_concurrent):
    scores = jury_concurrent.evaluate(predictions, references)

    assert all([scores[metric.resulting_name] is not None for metric in jury_concurrent.metrics])


def test_reduce_fn(predictions, references, jury):
    _reduce_fn = np.mean
    _non_reduce_fn = np.exp
    scores = jury.evaluate(predictions, references, reduce_fn=_reduce_fn)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])

    with pytest.raises(ValueError):
        jury.evaluate(predictions, references, reduce_fn=_non_reduce_fn)
