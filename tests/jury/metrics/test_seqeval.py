import pytest

from jury import Jury
from jury.metrics import AutoMetric
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    metric = AutoMetric.load("seqeval")
    return Jury(metrics=metric)


@pytest.fixture
def predictions():
    return [["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"], ["B-PER", "I-PER", "O"]]


@pytest.fixture
def references():
    return [["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"], ["B-PER", "I-PER", "O"]]


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


def test_basic(predictions, references, jury, output_basic):
    scores = jury(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)
