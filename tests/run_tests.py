from tests.utils import shell, validate_and_exit

if __name__ == "__main__":
    sts_test = shell("pytest --cov jury --cov-report term-missing --cov-report xml -vvv tests")
    validate_and_exit(sts_test)
