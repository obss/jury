import datasets
import numpy as np
import pytest

from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_concurrent():
    return output_evaluate_concurrent.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate():
    return output_evaluate.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_str_input():
    return output_evaluate_str_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_list_str_input():
    return output_evaluate_list_str_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_list_dict_input():
    return output_evaluate_list_dict_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_list_mixed_input():
    return output_evaluate_list_mixed_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_datasets_metric():
    return output_evaluate_datasets_metric.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_corpus():
    return output_evaluate_corpus.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_multiple_predictions():
    return output_evaluate_multiple_predictions.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_sequence_classification():
    return output_evaluate_sequence_classification.output


def test_evaluate_concurrent(predictions, references, jury_concurrent, output_evaluate_concurrent):
    scores = jury_concurrent(predictions=predictions, references=references)
    assert_almost_equal_dict(
        actual=scores, desired=output_evaluate_concurrent, exclude_paths="root['bertscore']['hashcode']"
    )


def test_evaluate_no_input(predictions, references, jury):
    with pytest.raises(TypeError):
        jury(predictions=predictions)
        jury(references=references)
        jury()


def test_evaluate_inconsistent_input(inconsistent_predictions, references, jury):
    # Different length
    with pytest.raises(ValueError):
        jury(predictions=inconsistent_predictions, references=references)


def test_evaluate_inconsistent_tasks(predictions, references, jury):
    with pytest.raises(ValueError):
        jury.add_metric("seqeval")
    jury.remove_metric("seqeval")


def test_evaluate(predictions, references, jury, output_evaluate):
    scores = jury(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate, exclude_paths="root['bertscore']['hashcode']")


def test_evaluate_str_input(predictions, references, jury_str, output_evaluate_str_input):
    scores = jury_str(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_str_input)


def test_evaluate_list_str_input(predictions, references, jury_list_str, output_evaluate_list_str_input):
    scores = jury_list_str(predictions=predictions, references=references)
    assert_almost_equal_dict(
        actual=scores, desired=output_evaluate_list_str_input, exclude_paths="root['bertscore']['hashcode']"
    )


def test_evaluate_list_dict_input(predictions, references, jury_list_dict, output_evaluate_list_dict_input):
    scores = jury_list_dict(predictions=predictions, references=references)
    assert_almost_equal_dict(
        actual=scores, desired=output_evaluate_list_dict_input, exclude_paths="root['bertscore']['hashcode']"
    )


def test_evaluate_list_mixed_input(predictions, references, jury_list_mixed, output_evaluate_list_mixed_input):
    scores = jury_list_mixed(predictions=predictions, references=references)
    assert_almost_equal_dict(
        actual=scores, desired=output_evaluate_list_mixed_input, exclude_paths="root['bertscore']['hashcode']"
    )


def test_evaluate_datasets_metric(predictions, references, jury_datasets, output_evaluate_datasets_metric):
    scores = jury_datasets(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_datasets_metric)


def test_evaluate_corpus(single_prediction_array, multiple_references, jury, output_evaluate_corpus):
    scores = jury(predictions=single_prediction_array, references=multiple_references)
    assert_almost_equal_dict(
        actual=scores, desired=output_evaluate_corpus, exclude_paths="root['bertscore']['hashcode']"
    )


def test_evaluate_multiple_predictions(
    multiple_predictions, multiple_references, jury, output_evaluate_multiple_predictions
):
    scores = jury(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(
        actual=scores, desired=output_evaluate_multiple_predictions, exclude_paths="root['bertscore']['hashcode']"
    )


def test_evaluate_sequence_classification(
    predictions_sequence_classification,
    references_sequence_classification,
    jury_sequence_classification,
    output_evaluate_sequence_classification,
):
    scores = jury_sequence_classification(
        predictions=predictions_sequence_classification, references=references_sequence_classification
    )
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_sequence_classification)


def test_reduce_fn(predictions, references, jury):
    _reduce_fn = np.mean
    _non_reduce_fn = np.exp
    scores = jury(predictions=predictions, references=references, reduce_fn=_reduce_fn)

    assert all([scores[metric.resulting_name] is not None for metric in jury.metrics])

    with pytest.raises(ValueError):
        jury(predictions=predictions, references=references, reduce_fn=_non_reduce_fn)


def test_load_metric():
    from jury import load_metric
    from jury.metrics._core import Metric as JuryMetric

    assert isinstance(load_metric("squad"), JuryMetric)
    assert isinstance(load_metric("squad_v2"), datasets.Metric)

    with pytest.raises(FileNotFoundError):
        load_metric("abcdefgh")
