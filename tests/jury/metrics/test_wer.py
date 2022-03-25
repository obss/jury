import pytest

from jury import Jury
from jury.metrics import AutoMetric
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury_wer():
    metric = AutoMetric.load("wer")
    return Jury(metrics=metric)


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_concat():
    return output_basic_concat.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref():
    return output_multiple_ref.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref():
    return output_multiple_pred_multiple_ref.output


def test_basic(predictions, references, jury_wer, output_basic):
    scores = jury_wer(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_basic_concat(predictions, references, jury_wer, output_basic_concat):
    scores = jury_wer(predictions=predictions, references=references, concatenate_texts=True)
    assert_almost_equal_dict(actual=scores, desired=output_basic_concat)


def test_multiple_ref(predictions, multiple_references, jury_wer, output_multiple_ref):
    scores = jury_wer(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_pred_multiple_ref(
    multiple_predictions, multiple_references, jury_wer, output_multiple_pred_multiple_ref
):
    scores = jury_wer(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)
