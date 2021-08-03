from jury import Jury
from jury.metrics.squad import SQUAD

METRICS = [SQUAD()]


def test_basic_str_input():
    _EXPECTED_RESULT = {
        "empty_predictions": 0,
        "total_items": 2,
        "SQUAD": 0.8333333333333333,
    }
    predictions = ["1917", "Albert Einstein"]
    references = ["1917", "Einstein"]

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT


def test_basic_dict_input():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 1, "SQUAD": 1.0}
    predictions = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
    references = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT


def test_squad_str_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "SQUAD": 1.0}
    predictions = ["1917", "Albert Einstein"]
    references = ["1917", ["Einstein", "Albert Einstein"]]

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT


def test_multiple_pred_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "SQUAD": 0.5}
    predictions = [["1917", "November 1917"], "Albert Einstein"]
    references = [["1917", "in November 1917"], ["Einstein", "Albert Einstein"]]

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT
