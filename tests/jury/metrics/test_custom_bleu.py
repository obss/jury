import pytest

from jury import Jury
from jury.metrics import AutoMetric
from tests.jury import TEST_DATA_DIR
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict

_CUSTOM_BLEU_PATH = TEST_DATA_DIR / "custom_bleu"


@pytest.fixture(scope="module")
def jury_custom_bleu():
    metric = AutoMetric.load(_CUSTOM_BLEU_PATH)
    return Jury(metrics=metric)


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


def test_basic(predictions, references, jury_custom_bleu, output_basic):
    scores = jury_custom_bleu(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_multiple_ref(predictions, multiple_references, jury_custom_bleu, output_multiple_ref):
    scores = jury_custom_bleu(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_pred_multiple_ref(
    multiple_predictions, multiple_references, jury_custom_bleu, output_multiple_pred_multiple_ref
):
    scores = jury_custom_bleu(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)
