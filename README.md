<h1 align="center">Jury</h1>

<p align="center">
<a href="https://github.com/obss/jury/actions"><img alt="Build status" src="https://github.com/obss/jury/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://badge.fury.io/py/jury"><img src="https://badge.fury.io/py/jury.svg" alt="PyPI version" height="20"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Simple tool/toolkit for evaluating NLG (Natural Language Generation) offering various automated metrics. Jury offers a smooth and easy-to-use interface. It uses huggingface/datasets package for underlying metric computation, and hence adding custom metric is easy as adopting `datasets.Metric`.

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

You can use custom metrics with inheriting `datasets.Metric`, you can see current metrics on [datasets/metrics](https://github.com/huggingface/datasets/tree/master/metrics). The code snippet below gives a brief explanation.

    import datasets

    CustomMetric(datasets.Metric):
        def _info(self):
            pass
        
        def _compute(self, predictions, references, *args, **kwargs):
            pass

## <div align="center"> Contributing </div>

PRs are welcomed as always :)

### Installation

    git clone https://github.com/obss/jury.git
    cd jury
    python setup.py develop
    pip install -r requirements-dev.txt

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
