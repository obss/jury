import pytest

from jury import Jury
from jury.metrics import AutoMetric
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury_prism():
    metric = AutoMetric.load("prism")
    return Jury(metrics=metric)


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_normalized_and_segmented():
    return output_basic_normalized_and_segmented.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref():
    return output_multiple_ref.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref_normalized_and_segmented():
    return output_multiple_ref_normalized_and_segmented.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref():
    return output_multiple_pred_multiple_ref.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref_normalized_and_segmented():
    return output_multiple_pred_multiple_ref_normalized_and_segmented.output


def test_basic(predictions, references, jury_prism, output_basic):
    scores = jury_prism(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_basic_normalized_and_segmented(predictions, references, jury_prism, output_basic_normalized_and_segmented):
    scores = jury_prism(predictions=predictions, references=references, normalize=True, segment_scores=True)
    assert_almost_equal_dict(actual=scores, desired=output_basic_normalized_and_segmented)


def test_multiple_ref(predictions, multiple_references, jury_prism, output_multiple_ref):
    scores = jury_prism(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_ref_normalized_and_segmented(
    predictions, references, jury_prism, output_multiple_ref_normalized_and_segmented
):
    scores = jury_prism(predictions=predictions, references=references, normalize=True, segment_scores=True)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref_normalized_and_segmented)


def test_multiple_pred_multiple_ref(
    multiple_predictions, multiple_references, jury_prism, output_multiple_pred_multiple_ref
):
    scores = jury_prism(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)


def test_multiple_pred_multiple_ref_normalized_and_segmented(
    predictions,
    references,
    jury_prism,
    output_multiple_pred_multiple_ref_normalized_and_segmented,
):
    scores = jury_prism(predictions=predictions, references=references, normalize=True, segment_scores=True)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref_normalized_and_segmented)
