import datasets
import numpy as np
import pytest


def test_evaluate_concurrent(predictions, references, jury_concurrent):
    scores = jury_concurrent(predictions=predictions, references=references)

    assert all([scores[metric.resulting_name] is not None for metric in jury_concurrent.metrics])


def test_evaluate_no_input(predictions, references, jury):
    with pytest.raises(TypeError):
        jury(predictions=predictions)
        jury(references=references)
        jury()


def test_evaluate_inconsistent_input(inconsistent_predictions, references, jury):
    # Different length
    with pytest.raises(ValueError):
        jury(predictions=inconsistent_predictions, references=references)


def test_evaluate_basic(predictions, references, jury):
    scores = jury(predictions=predictions, references=references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_basic_str_input(predictions, references, jury_str):
    scores = jury_str(predictions=predictions, references=references)

    assert all([scores[metric.resulting_name] is not None for metric in jury_str.metrics])


def test_evaluate_datasets_metric(predictions, references, jury_datasets):
    scores = jury_datasets(predictions=predictions, references=references)

    assert all([scores[metric.resulting_name] is not None for metric in jury_datasets.metrics])


def test_evaluate_corpus(single_prediction_array, multiple_references, jury):
    scores = jury(predictions=single_prediction_array, references=multiple_references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_evaluate_multiple_predictions(multiple_predictions, multiple_references, jury):
    scores = jury(predictions=multiple_predictions, references=multiple_references)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])


def test_reduce_fn(predictions, references, jury):
    _reduce_fn = np.mean
    _non_reduce_fn = np.exp
    scores = jury(predictions=predictions, references=references, reduce_fn=_reduce_fn)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])

    with pytest.raises(ValueError):
        jury(predictions=predictions, references=references, reduce_fn=_non_reduce_fn)


def test_load_metric():
    from jury import load_metric
    from jury.metrics import Metric as JuryMetric

    assert isinstance(load_metric("squad"), JuryMetric)
    assert isinstance(load_metric("squad_v2"), datasets.Metric)

    with pytest.raises(FileNotFoundError):
        load_metric("abcdefgh")
