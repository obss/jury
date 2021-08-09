<h1 align="center">Jury</h1>

<p align="center">
<a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/pyversions/jury" alt="Python versions"></a>
<a href="https://pepy.tech/project/jury"><img src="https://pepy.tech/badge/jury" alt="downloads"></a>
<a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/v/jury?color=blue" alt="PyPI version"></a>
<a href="https://github.com/obss/jury/releases/latest"><img alt="Latest Release" src="https://img.shields.io/github/release-date/obss/jury"></a>
<br>
<a href="https://github.com/obss/jury/actions"><img alt="Build status" src="https://github.com/obss/jury/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://libraries.io/pypi/jury"><img alt="Dependencies" src="https://img.shields.io/librariesio/github/obss/jury"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/obss/jury/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/jury"></a>
</p>

Simple tool/toolkit for evaluating NLG (Natural Language Generation) offering various automated metrics. Jury offers a smooth and easy-to-use interface. It uses [datasets](https://github.com/huggingface/datasets/) for underlying metric computation, and hence adding custom metric is easy as adopting `datasets.Metric`. 

Main advantages that Jury offers are:

- Easy to use for any NLG system.
- Calculate many metrics at once.
- Metrics calculations are handled concurrently to save processing time.
- It supports evaluating multiple predictions.

To see more, check the [official Jury blog post](https://medium.com/codable/jury-evaluating-performance-of-nlg-models-730eb9c9999f).

## <div align="center"> Installation </div>

Through pip,

    pip install jury

or build from source,

    git clone https://github.com/obss/jury.git
    cd jury
    python setup.py install

## <div align="center"> Usage </div>

### API Usage

It is only two lines of code to evaluate generated outputs.

    from jury import Jury
    
    jury = Jury()

    # Microsoft translator translation for "Yurtta sulh, cihanda sulh." (16.07.2021)
    predictions = ["Peace in the dormitory, peace in the world."]
    references = ["Peace at home, peace in the world."]
    scores = jury.evaluate(predictions, references)

Specify metrics you want to use on instantiation.

    jury = Jury(metrics=["bleu", "meteor"])
    scores = jury.evaluate(predictions, references)

### Custom Metrics

You can use custom metrics with inheriting `jury.metrics.Metric`, you can see current metrics on [datasets/metrics](https://github.com/huggingface/datasets/tree/master/metrics). The code snippet below gives a brief explanation.

    from jury.metrics import Metric

    CustomMetric(Metric):
        def compute(self, predictions, references):
            pass

## <div align="center"> Contributing </div>

PRs are welcomed as always :)

### Installation

    git clone https://github.com/obss/jury.git
    cd jury
    pip install -e .[develop]

### Tests

To tests simply run.

    python tests/run_tests.py

### Code Style

To check code style,

    python tests/run_code_style.py check

To format codebase,

    python tests/run_code_style.py format


## <div align="center"> License </div>

Licensed under the [MIT](LICENSE) License.
