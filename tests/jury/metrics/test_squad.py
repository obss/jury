import pytest

from jury import Jury
from jury.metrics.squad import Squad
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    return Jury(metrics=[Squad()])


@pytest.fixture
def predictions():
    return ["1917", "Albert Einstein"]


@pytest.fixture
def references():
    return ["1917", "Einstein"]


@pytest.fixture
def multiple_references():
    return [["1917"], ["Einstein", "Albert Einstein"]]


@pytest.fixture
def multiple_predictions():
    return [["1917", "November 1917"], ["Albert Einstein"]]


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref():
    return output_multiple_ref.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref():
    return output_multiple_pred_multiple_ref.output


def test_basic(predictions, references, jury, output_basic):
    scores = jury.evaluate(predictions, references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_multiple_ref(predictions, multiple_references, jury, output_multiple_ref):
    scores = jury.evaluate(predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury, output_multiple_pred_multiple_ref):
    scores = jury.evaluate(multiple_predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)
