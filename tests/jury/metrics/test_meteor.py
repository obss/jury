from jury import Jury
from jury.metrics.meteor import Meteor
from tests.jury import _DEFAULT_PREDICTIONS, _DEFAULT_PREDICTIONS_MR, _DEFAULT_REFERENCES, _DEFAULT_REFERENCES_MR

METRICS = [Meteor()]


def test_basic():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "Meteor": 0.6594164989939638}
    predictions = _DEFAULT_PREDICTIONS
    references = _DEFAULT_REFERENCES

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT


def test_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "Meteor": 0.6594164989939638}
    predictions = _DEFAULT_PREDICTIONS
    references = _DEFAULT_REFERENCES_MR

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT


def test_multiple_pred_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "Meteor": 0.6915697081036947}
    predictions = _DEFAULT_PREDICTIONS_MR
    references = _DEFAULT_REFERENCES_MR

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert scores == _EXPECTED_RESULT
