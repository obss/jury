import pytest

from jury import Jury
from jury.metrics.bertscore import Bertscore
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def jury():
    return Jury(metrics=[Bertscore(params={"model_type": "albert-base-v1", "device": "cpu"})])


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


def test_basic(predictions, references, jury, output_basic):
    scores = jury.evaluate(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic, exclude_paths="root['bertscore']['hashcode']")


def test_multiple_ref(predictions, multiple_references, jury, output_multiple_ref):
    scores = jury.evaluate(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref, exclude_paths="root['bertscore']['hashcode']")


def test_multiple_pred_multiple_ref(multiple_predictions, multiple_references, jury, output_multiple_pred_multiple_ref):
    scores = jury.evaluate(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(
        actual=scores, desired=output_multiple_pred_multiple_ref, exclude_paths="root['bertscore']['hashcode']"
    )
