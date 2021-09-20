import json
import os
import re
import sys
from typing import Dict

from deepdiff import DeepDiff


def shell(command, exit_status=0):
    """
    Run command through shell and return exit status if exit status of command run match with given exit status.

    Args:
        command: (str) Command string which runs through system shell.
        exit_status: (int) Expected exit status of given command run.

    Returns: actual_exit_status

    """
    actual_exit_status = os.system(command)
    if actual_exit_status == exit_status:
        return 0
    return actual_exit_status


def validate_and_exit(*args, expected_out_status=0):
    if all([arg == expected_out_status for arg in args]):
        # Expected status, OK
        sys.exit(0)
    else:
        # Failure
        sys.exit(1)


def assert_almost_equal_dict(actual: Dict, desired: Dict, decimal=5):
    assert DeepDiff(actual, desired, significant_digits=decimal) == {}


def shell_capture(command, out_json=True):
    out = os.popen(command).read()
    if out_json:
        out = re.findall(r"{\s+.*\}", out, flags=re.MULTILINE | re.DOTALL)[0].replace("\n", "")
        return json.loads(out)
    return out
