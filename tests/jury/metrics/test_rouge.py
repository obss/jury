import pytest

from jury import Jury
from jury.metrics.rouge import Rouge
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    return Jury(metrics=[Rouge()])


def test_basic(predictions, references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "rougeL": 0.6190476190476191}

    scores = jury.evaluate(predictions, references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)


def test_multiple_ref(predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "rougeL": 0.6190476190476191}

    scores = jury.evaluate(predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "rougeL": 0.7948717948717948}

    scores = jury.evaluate(multiple_predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)
