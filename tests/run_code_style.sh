#!/bin/bash

set -xeu

if [ $1 = "check" ]; then
    # stop the build if there are Python syntax errors or undefined names
    flake8 jury tests --config setup.cfg
    black . --check --config pyproject.toml
elif [ $1 = "format" ]; then
    black . --config pyproject.toml
fi