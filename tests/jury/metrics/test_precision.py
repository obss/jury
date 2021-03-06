import pytest

from jury import Jury
from jury.metrics import AutoMetric
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury_precision_language_generation():
    metric = AutoMetric.load("precision")
    return Jury(metrics=metric)


@pytest.fixture(scope="module")
def jury_precision_sequence_classification():
    metric = AutoMetric.load("precision", task="sequence-classification")
    return Jury(metrics=metric)


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_language_generation():
    return output_basic_language_generation.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref_language_generation():
    return output_multiple_ref_language_generation.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref_language_generation():
    return output_multiple_pred_multiple_ref_language_generation.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_sequence_classification():
    return output_basic_sequence_classification.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref_sequence_classification():
    return output_multiple_ref_sequence_classification.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref_sequence_classification():
    return output_multiple_pred_multiple_ref_sequence_classification.output


def test_basic_language_generation(
    predictions, references, jury_precision_language_generation, output_basic_language_generation
):
    scores = jury_precision_language_generation(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic_language_generation)


def test_multiple_ref_language_generation(
    predictions, multiple_references, jury_precision_language_generation, output_multiple_ref_language_generation
):
    scores = jury_precision_language_generation(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref_language_generation)


def test_multiple_pred_multiple_ref_language_generation(
    multiple_predictions,
    multiple_references,
    jury_precision_language_generation,
    output_multiple_pred_multiple_ref_language_generation,
):
    scores = jury_precision_language_generation(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref_language_generation)


def test_basic_sequence_classification(
    predictions_sequence_classification,
    references_sequence_classification,
    jury_precision_sequence_classification,
    output_basic_sequence_classification,
):
    scores = jury_precision_sequence_classification(
        predictions=predictions_sequence_classification, references=references_sequence_classification
    )
    assert_almost_equal_dict(actual=scores, desired=output_basic_sequence_classification)


def test_multiple_ref_sequence_classification(
    predictions_sequence_classification,
    multiple_references_sequence_classification,
    jury_precision_sequence_classification,
    output_multiple_ref_sequence_classification,
):
    scores = jury_precision_sequence_classification(
        predictions=predictions_sequence_classification, references=multiple_references_sequence_classification
    )
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref_sequence_classification)


def test_multiple_pred_multiple_ref_sequence_classification(
    multiple_predictions_sequence_classification,
    multiple_references_sequence_classification,
    jury_precision_sequence_classification,
    output_multiple_pred_multiple_ref_sequence_classification,
):
    scores = jury_precision_sequence_classification(
        predictions=multiple_predictions_sequence_classification, references=multiple_references_sequence_classification
    )
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref_sequence_classification)
