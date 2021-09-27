from pathlib import Path

import pytest

from tests.jury import TEST_DATA_DIR
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict, shell_capture


@pytest.fixture
@get_expected_output(prefix=None)
def output_cli():
    return output_cli.output


def test_cli(output_cli):
    predictions_file = Path(TEST_DATA_DIR) / "cli" / "predictions.txt"
    references_file = Path(TEST_DATA_DIR) / "cli" / "references.txt"

    cmd = f"jury eval --predictions {predictions_file} --references {references_file}"

    out = shell_capture(cmd)
    assert_almost_equal_dict(actual=out, desired=output_cli)
