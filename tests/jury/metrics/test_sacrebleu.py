import pytest

from jury import Jury
from jury.metrics.sacrebleu import SacreBLEU
from tests.utils import assert_almost_equal_dict


@pytest.fixture
def jury():
    return Jury(metrics=[SacreBLEU()])


def test_basic(predictions, references, jury):
    _EXPECTED_RESULT = {"SacreBLEU": 0.4765798792330085, "empty_predictions": 0, "total_items": 2}

    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_multiple_ref(predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"SacreBLEU": 0.4765798792330085, "empty_predictions": 0, "total_items": 2}

    scores = jury.evaluate(predictions, multiple_references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "SacreBLEU": 0.4961395531582874}

    scores = jury.evaluate(multiple_predictions, multiple_references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)
