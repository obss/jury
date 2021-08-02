import sys

from tests.utils import assert_shell

if __name__ == "__main__":
    arg = sys.argv[1]

    if arg == "check":
        assert_shell("flake8 jury tests --config setup.cfg")
        assert_shell("isort . --check --settings setup.cfg")
        assert_shell("black . --check --config pyproject.toml")
    elif arg == "format":
        assert_shell("isort . --settings setup.cfg")
        assert_shell("black . --config pyproject.toml")
