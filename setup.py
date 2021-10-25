import io
import os
import platform
import re
from typing import List

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "jury", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def add_pywin(reqs: List[str]) -> None:
    if platform.system() == "Windows":
        # Latest PyWin32 build (301) fails, required for sacrebleu
        ext_package = ["pywin32==228"]
    else:
        ext_package = []
    reqs.extend(ext_package)


_DEV_REQUIREMENTS = [
    "black==21.7b0",
    "deepdiff==5.5.0",
    "flake8==3.9.2",
    "isort==5.9.2",
    "jiwer>=2.2.0",
    "pytest>=6.2.4",
    "pytest-cov>=2.12.1",
    "pytest-timeout>=1.4.2",
]

_METRIC_REQUIREMENTS = ["sacrebleu>=2.0.0", "bert_score==0.3.10", "seqeval==1.2.2"]
add_pywin(_METRIC_REQUIREMENTS)

extras = {
    "metrics": _METRIC_REQUIREMENTS,
    "dev": _DEV_REQUIREMENTS + _METRIC_REQUIREMENTS,
}


setuptools.setup(
    name="jury",
    version=get_version(),
    author="",
    license="MIT",
    description="Evaluation toolkit for neural language generation.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/obss/jury",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    extras_require=extras,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "jury=jury.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning, deep-learning, ml, pytorch, NLP, evaluation, question-answering, question-generation",
)
