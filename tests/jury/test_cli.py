from pathlib import Path

import pytest

from tests.jury import TEST_DATA_DIR
from tests.jury.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict, shell_capture


@pytest.fixture
@get_expected_output(prefix=None)
def output_cli_from_file():
    return output_cli_from_file.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_cli_from_folder():
    return output_cli_from_folder.output


def test_cli_from_file(output_cli_from_file):
    predictions_file = str(Path(TEST_DATA_DIR) / "cli" / "from_file" / "predictions.txt")
    references_file = str(Path(TEST_DATA_DIR) / "cli" / "from_file" / "references.txt")

    cmd = f"jury eval --predictions {predictions_file} --references {references_file}"
    out = shell_capture(cmd)
    assert_almost_equal_dict(actual=out, desired=output_cli_from_file)


def test_cli_from_folder(output_cli_from_folder):
    predictions_folder = str(Path(TEST_DATA_DIR) / "cli" / "from_folder" / "predictions")
    references_folder = str(Path(TEST_DATA_DIR) / "cli" / "from_folder" / "references")

    cmd = f"jury eval --predictions {predictions_folder} --references {references_folder}"
    out = shell_capture(cmd)
    assert_almost_equal_dict(actual=out, desired=output_cli_from_folder)
