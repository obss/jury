from jury import Jury
from jury.metrics.sacrebleu import SacreBLEU
from tests.jury import _DEFAULT_PREDICTIONS, _DEFAULT_PREDICTIONS_MR, _DEFAULT_REFERENCES, _DEFAULT_REFERENCES_MR
from tests.utils import assert_almost_equal_dict

METRICS = [SacreBLEU()]


def test_basic():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "SacreBLEU": 0.4599439537698078}
    predictions = _DEFAULT_PREDICTIONS
    references = _DEFAULT_REFERENCES

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "SacreBLEU": 0.4599439537698078}
    predictions = _DEFAULT_PREDICTIONS
    references = _DEFAULT_REFERENCES_MR

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_multiple_pred_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "SacreBLEU": 0.4961395531582874}
    predictions = _DEFAULT_PREDICTIONS_MR
    references = _DEFAULT_REFERENCES_MR

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)
