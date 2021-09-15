import pytest

from jury import Jury
from jury.metrics.bertscore import BERTScore
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    return Jury(metrics=[BERTScore(params={"model_type": "albert-base-v1"})])


def test_basic(predictions, references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "BERTScore": 0.8293454647064209}

    scores = jury.evaluate(predictions, references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)


def test_multiple_ref(predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "BERTScore": 0.7350329756736755}

    scores = jury.evaluate(predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "BERTScore": 0.7431023120880127}

    scores = jury.evaluate(multiple_predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)
