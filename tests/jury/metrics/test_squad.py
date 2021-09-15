import pytest

from jury import Jury
from jury.metrics.squad import SQUAD_F1
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    return Jury(metrics=[SQUAD_F1()])


@pytest.fixture
def predictions():
    return ["1917", "Albert Einstein"]


@pytest.fixture
def references():
    return ["1917", "Einstein"]


@pytest.fixture
def multiple_references():
    return ["1917", ["Einstein", "Albert Einstein"]]


@pytest.fixture
def multiple_predictions():
    return [["1917", "November 1917"], "Albert Einstein"]


@pytest.fixture()
def squad_style_predictions():
    return [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]


@pytest.fixture
def squad_style_references():
    return [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]


def test_basic_str_input(predictions, references, jury):
    _EXPECTED_RESULT = {
        "empty_predictions": 0,
        "total_items": 2,
        "squad_f1": 0.8333333333333333,
    }

    scores = jury.evaluate(predictions, references)
    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_basic_dict_input(squad_style_predictions, squad_style_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 1, "squad_f1": 1.0}

    scores = jury.evaluate(squad_style_predictions, squad_style_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)


def test_multiple_ref(predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "squad_f1": 1.0}

    scores = jury.evaluate(predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury):
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "squad_f1": 0.5}

    scores = jury.evaluate(multiple_predictions, multiple_references)
    assert_almost_equal_dict(actual=scores, desired=_EXPECTED_RESULT)
