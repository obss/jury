import json
import os
import re
from typing import Dict

from deepdiff import DeepDiff


def assert_shell(command, exit_status=0):
    """
    Run command through shell and return exit status if exit status of command run match with given exit status,
    raise an error otherwise.

    Args:
        command: (str) Command string which runs through system shell.
        exit_status: (int) Expected exit status of given command run.

    Returns: actual_exit_status
    """
    actual_exit_status = os.system(command)
    assert exit_status == actual_exit_status, f"Unexpected exit code {str(actual_exit_status)}"
    return actual_exit_status


def assert_almost_equal_dict(actual: Dict, desired: Dict, decimal=5):
    assert DeepDiff(actual, desired, significant_digits=decimal + 1) == {}


def shell_capture(command, out_json=True):
    out = os.popen(command).read()
    if out_json:
        out = re.findall(r"{\s+.*\}", out, flags=re.MULTILINE | re.DOTALL)[0].replace("\n", "")
        return json.loads(out)
    return out
