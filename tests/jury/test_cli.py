from pathlib import Path

import pytest

from jury import TEST_DATA_DIR
from tests.utils import assert_almost_equal_dict, shell_capture


@pytest.fixture
def desired_output():
    return {
        "empty_predictions": 0,
        "total_items": 2,
        "bleu_1": 0.6666666666666666,
        "bleu_2": 0.5063696835418333,
        "bleu_3": 0.4119912453171404,
        "bleu_4": 0.29689669509442307,
        "Meteor": 0.6594164989939638,
        "rougeL": 0.6190476190476191,
        "SacreBLEU": 0.4765798792330085,
    }


def test_cli(desired_output):
    predictions_file = Path(TEST_DATA_DIR) / "predictions.txt"
    references_file = Path(TEST_DATA_DIR) / "references.txt"

    cmd = f"jury eval --predictions {predictions_file} --references {references_file}"

    out = shell_capture(cmd)
    assert_almost_equal_dict(desired_output, out)
