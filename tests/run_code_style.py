import os
import sys

if __name__ == "__main__":
    arg = sys.argv[1]

    if arg == "check":
        os.system("flake8 jury tests --config setup.cfg")
        os.system("isort . --check --settings setup.cfg")
        os.system("black . --check --config pyproject.toml")
    elif arg == "format":
        os.system("isort . --settings setup.cfg")
        os.system("black . --config pyproject.toml")
