<h1 align="center">Jury</h1>

<p align="center">
<a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/pyversions/jury" alt="Python versions"></a>
<a href="https://pepy.tech/project/jury"><img src="https://pepy.tech/badge/jury" alt="downloads"></a>
<a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/v/jury?color=blue" alt="PyPI version"></a>
<a href="https://github.com/obss/jury/releases/latest"><img alt="Latest Release" src="https://img.shields.io/github/release-date/obss/jury"></a>
<a href="https://colab.research.google.com/github/obss/jury/blob/main/examples/jury_evaluate.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

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

```python
from jury import Jury

scorer = Jury()
predictions = [
    ["the cat is on the mat", "There is cat playing on the mat"], 
    ["Look!    a wonderful day."]
]
references = [
    ["the cat is playing on the mat.", "The cat plays on the mat."], 
    ["Today is a wonderful day", "The weather outside is wonderful."]
]
scores = scorer(predictions=predictions, references=references)
```

Specify metrics you want to use on instantiation.

```python
scorer = Jury(metrics=["bleu", "meteor"])
scores = scorer(predictions, references)
```

#### Use of Metrics standalone

You can directly import metrics from `jury.metrics` as classes, and then instantiate and use as desired.

```python
from jury.metrics import Bleu

bleu = Bleu.construct()
score = bleu.compute(predictions=predictions, references=references)
```

The additional parameters can either be specified on `compute()`

```python
from jury.metrics import Bleu

bleu = Bleu.construct()
score = bleu.compute(predictions=predictions, references=references, max_order=4)
```

, or alternatively on instantiation

```python
bleu = Bleu._construct(compute_kwargs={"max_order": 1})
```

Note that you can seemlessly access both `jury` and `datasets` metrics through `jury.load_metric`. 

```python
import jury

bleu = jury.load_metric("bleu")
bleu_1 = jury.load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1})
# metrics not available in `jury` but in `datasets`
wer = jury.load_metric("wer") # It falls back to `datasets` package with a warning
```

### CLI Usage

You can specify predictions file and references file paths and get the resulting scores. Each line should be paired in both files.

    jury eval --predictions /path/to/predictions.txt --references /path/to/references.txt --reduce_fn max

If you want to specify metrics, and do not want to use default, specify it in config file (json) in `metrics` key.

```json
{
  "predictions": "/path/to/predictions.txt",
  "references": "/path/to/references.txt",
  "reduce_fn": "max",
  "metrics": [
    "bleu",
    "meteor"
  ]
}
```

Then, you can call jury eval with `config` argument.

    jury eval --config path/to/config.json

### Custom Metrics

You can use custom metrics with inheriting `jury.metrics.Metric`, you can see current metrics implemented on Jury from [jury/metrics](https://github.com/obss/jury/tree/master/jury/metrics). Jury falls back to `datasets` implementation of metrics for the ones that are currently not supported by Jury, you can see the metrics available for `datasets` on [datasets/metrics](https://github.com/huggingface/datasets/tree/master/metrics). 

Jury itself uses `datasets.Metric` as a base class to drive its own base class as `jury.metrics.Metric`. The interface is similar; however, Jury makes the metrics to take a unified input type by handling the inputs for each metrics, and allows supporting several input types as;

- single prediction & single reference
- single prediction & multiple reference
- multiple prediction & multiple reference

As a custom metric both base classes can be used; however, we strongly recommend using `jury.metrics.Metric` as it has several advantages such as supporting computations for the input types above or unifying the type of the input.

```python
from jury.metrics import MetricForTask

class CustomMetric(MetricForTask):
    def _compute_single_pred_single_ref(
        self, predictions, references, reduce_fn = None, **kwargs
    ):
        raise NotImplementedError

    def _compute_single_pred_multi_ref(
        self, predictions, references, reduce_fn = None, **kwargs
    ):
        raise NotImplementedError

    def _compute_multi_pred_multi_ref(
            self, predictions, references, reduce_fn = None, **kwargs
    ):
        raise NotImplementedError
```

For more details, have a look at base metric implementation [jury.metrics.Metric](./jury/metrics/_base.py)

## <div align="center"> Contributing </div>

PRs are welcomed as always :)

### Installation

    git clone https://github.com/obss/jury.git
    cd jury
    pip install -e .[dev]

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
