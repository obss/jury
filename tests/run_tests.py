import os


if __name__ == "__main__":
    os.system("pytest --cov jury --cov-report term-missing --cov-report xml -vvv tests")
