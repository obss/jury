from jury import Jury
from jury.metrics.bleu import BLEU
from tests.jury import _DEFAULT_PREDICTIONS, _DEFAULT_PREDICTIONS_MR, _DEFAULT_REFERENCES, _DEFAULT_REFERENCES_MR
from tests.utils import assert_almost_equal_dict

METRICS = [BLEU()]


def test_basic():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "BLEU": 0.18277761142725615}
    predictions = _DEFAULT_PREDICTIONS
    references = _DEFAULT_REFERENCES

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "BLEU": 0.18277761142725615}
    predictions = _DEFAULT_PREDICTIONS
    references = _DEFAULT_REFERENCES_MR

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)


def test_multiple_pred_multiple_ref():
    _EXPECTED_RESULT = {"empty_predictions": 0, "total_items": 2, "BLEU": 0.410274681629658}
    predictions = _DEFAULT_PREDICTIONS_MR
    references = _DEFAULT_REFERENCES_MR

    jury = Jury(metrics=METRICS)
    scores = jury.evaluate(predictions, references)

    assert_almost_equal_dict(_EXPECTED_RESULT, scores)
