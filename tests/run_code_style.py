import sys

from tests.utils import shell, validate_and_exit

if __name__ == "__main__":
    arg = sys.argv[1]

    if arg == "check":
        sts_flake = shell("flake8 jury tests --config setup.cfg")
        sts_isort = shell("isort . --check --settings setup.cfg")
        sts_black = shell("black . --check --config pyproject.toml")
        validate_and_exit(flake8=sts_flake, isort=sts_isort, black=sts_black)
    elif arg == "format":
        sts_isort = shell("isort . --settings setup.cfg")
        sts_black = shell("black . --config pyproject.toml")
        validate_and_exit(isort=sts_isort, black=sts_black)
