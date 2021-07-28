#!/bin/bash

set -xeu

pytest --cov jury --cov-report term-missing --cov-report xml -vvv tests