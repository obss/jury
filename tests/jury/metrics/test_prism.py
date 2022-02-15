import pytest

from jury import Jury
from jury.metrics import AutoMetric
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    metric = AutoMetric.load("prism")
    return Jury(metrics=metric)


@pytest.fixture(scope="module")
def jury_normalized():
    metric = AutoMetric.load("prism", compute_kwargs={"normalize": True})
    return Jury(metrics=metric)


@pytest.fixture(scope="module")
def jury_segmented():
    metric = AutoMetric.load("prism", compute_kwargs={"segment_scores": True})
    return Jury(metrics=metric)


@pytest.fixture(scope="module")
def jury_normalized_and_segmented():
    metric = AutoMetric.load("prism", compute_kwargs={"normalize": True, "segment_scores": True})
    return Jury(metrics=metric)


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_normalized():
    return output_basic_normalized.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_segmented():
    return output_basic_segmented.output


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
def output_multiple_ref_normalized():
    return output_multiple_ref_normalized.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref_segmented():
    return output_multiple_ref_segmented.output


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
def output_multiple_pred_multiple_ref_normalized():
    return output_multiple_pred_multiple_ref_normalized.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref_segmented():
    return output_multiple_pred_multiple_ref_segmented.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref_normalized_and_segmented():
    return output_multiple_pred_multiple_ref_normalized_and_segmented.output


def test_basic(predictions, references, jury, output_basic):
    scores = jury(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_basic_normalized(predictions, references, jury_normalized, output_basic_normalized):
    scores = jury_normalized(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic_normalized)


def test_basic_segmented(predictions, references, jury_segmented, output_basic_segmented):
    scores = jury_segmented(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic_segmented)


def test_basic_normalized_and_segmented(
    predictions, references, jury_normalized_and_segmented, output_basic_normalized_and_segmented
):
    scores = jury_normalized_and_segmented(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic_normalized_and_segmented)


def test_multiple_ref(predictions, multiple_references, jury, output_multiple_ref):
    scores = jury(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_ref_normalized(predictions, references, jury_normalized, output_multiple_ref_normalized):
    scores = jury_normalized(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref_normalized)


def test_multiple_ref_segmented(predictions, references, jury_segmented, output_multiple_ref_segmented):
    scores = jury_segmented(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref_segmented)


def test_multiple_ref_normalized_and_segmented(
    predictions, references, jury_normalized_and_segmented, output_multiple_ref_normalized_and_segmented
):
    scores = jury_normalized_and_segmented(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref_normalized_and_segmented)


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury, output_multiple_pred_multiple_ref):
    scores = jury(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)


def test_multiple_pred_multiple_ref_normalized(
    multiple_predictions, multiple_references, jury_normalized, output_multiple_pred_multiple_ref_normalized
):
    scores = jury_normalized(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref_normalized)


def test_multiple_pred_multiple_ref_segmented(
    predictions, references, jury_segmented, output_multiple_pred_multiple_ref_segmented
):
    scores = jury_segmented(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref_segmented)


def test_multiple_pred_multiple_ref_normalized_and_segmented(
    predictions, references, jury_normalized_and_segmented, output_multiple_pred_multiple_ref_normalized_and_segmented
):
    scores = jury_normalized_and_segmented(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref_normalized_and_segmented)
