<h1 align="center">Jury</h1>

<p align="center">
<a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/pyversions/jury" alt="Python versions"></a>
<a href="https://pepy.tech/project/jury"><img src="https://pepy.tech/badge/jury" alt="downloads"></a>
<a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/v/jury?color=blue" alt="PyPI version"></a>
<a href="https://github.com/obss/jury/releases/latest"><img alt="Latest Release" src="https://img.shields.io/github/release-date/obss/jury"></a>
<a href="https://colab.research.google.com/github/obss/jury/blob/main/examples/jury_evaluate.ipynb" target="_blank"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<br>
<a href="https://github.com/obss/jury/actions"><img alt="Build status" src="https://github.com/obss/jury/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://libraries.io/pypi/jury"><img alt="Dependencies" src="https://img.shields.io/librariesio/github/obss/jury"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/obss/jury/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/jury"></a>
<br>
<a href="https://doi.org/10.5281/zenodo.6109838"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6109838.svg" alt="DOI"></a>
</p>

A comprehensive toolkit for evaluating NLP experiments offering various automated metrics. Jury offers a smooth and easy-to-use interface. It uses a more advanced version of [evaluate](https://github.com/huggingface/evaluate/) design for underlying metric computation, so that adding custom metric is easy as extending proper class.

Main advantages that Jury offers are:

- Easy to use for any NLP project.
- Unified structure for computation input across all metrics.
- Calculate many metrics at once.
- Metrics calculations can be handled concurrently to save processing time.
- It seamlessly supports evaluation for multiple predictions/multiple references.

To see more, check the [official Jury blog post](https://medium.com/codable/jury-evaluating-performance-of-nlg-models-730eb9c9999f).

# Available Metrics

The table below shows the current support status for available metrics.

| Metric                                                                        | Jury Support       | HF/evaluate Support |
|-------------------------------------------------------------------------------|--------------------|---------------------|
| Accuracy-Numeric                                                              | :heavy_check_mark: | :white_check_mark:  |
| Accuracy-Text                                                                 | :heavy_check_mark: | :x:                 |
| Bartscore                                                                     | :heavy_check_mark: | :x:                 |
| Bertscore                                                                     | :heavy_check_mark: | :white_check_mark:  |
| Bleu                                                                          | :heavy_check_mark: | :white_check_mark:  |
| Bleurt                                                                        | :heavy_check_mark: | :white_check_mark:  |
| CER                                                                           | :heavy_check_mark: | :white_check_mark:  |
| CHRF                                                                          | :heavy_check_mark: | :white_check_mark:  |
| COMET                                                                         | :heavy_check_mark: | :white_check_mark:  |
| F1-Numeric                                                                    | :heavy_check_mark: | :white_check_mark:  |
| F1-Text                                                                       | :heavy_check_mark: | :x:                 |
| METEOR                                                                        | :heavy_check_mark: | :white_check_mark:  |
| Precision-Numeric                                                             | :heavy_check_mark: | :white_check_mark:  |
| Precision-Text                                                                | :heavy_check_mark: | :x:                 |
| Prism                                                                         | :heavy_check_mark: | :x:                 |
| Recall-Numeric                                                                | :heavy_check_mark: | :white_check_mark:  |
| Recall-Text                                                                   | :heavy_check_mark: | :x:                 |
| ROUGE                                                                         | :heavy_check_mark: | :white_check_mark:  |
| SacreBleu                                                                     | :heavy_check_mark: | :white_check_mark:  |
| Seqeval                                                                       | :heavy_check_mark: | :white_check_mark:  |
| Squad                                                                         | :heavy_check_mark: | :white_check_mark:  |
| TER                                                                           | :heavy_check_mark: | :white_check_mark:  |
| WER                                                                           | :heavy_check_mark: | :white_check_mark:  |
| [Other metrics](https://github.com/huggingface/evaluate/tree/master/metrics)* | :white_check_mark: | :white_check_mark:  |

_*_ Placeholder for the rest of the metrics available in `evaluate` package apart from those which are present in the 
table. 

**Notes**

* The entry :heavy_check_mark: represents that full Jury support is available meaning that all combinations of input 
types (single prediction & single reference, single prediction & multiple references, multiple predictions & multiple 
references) are supported

* The entry :white_check_mark: means that this metric is supported (for Jury through the `evaluate`), so that it 
can (and should) be used just like the `evaluate` metric as instructed in `evaluate` implementation although 
unfortunately full Jury support for those metrics are not yet available.

## Request for a New Metric

For the request of a new metric please [open an issue](https://github.com/obss/jury/issues/new?assignees=&labels=&template=new-metric.md&title=) providing the minimum information. Also, PRs addressing new metric 
supports are welcomed :).

## <div align="center"> Installation </div>

Through pip,

    pip install jury

or build from source,

    git clone https://github.com/obss/jury.git
    cd jury
    python setup.py install

**NOTE:** There may be malfunctions of some metrics depending on `sacrebleu` package on Windows machines which is 
mainly due to the package `pywin32`. For this, we fixed pywin32 version on our setup config for Windows platforms. 
However, if pywin32 causes trouble in your environment we strongly recommend using `conda` manager install the package 
as `conda install pywin32`.

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
from jury.metrics import Bleu
bleu = Bleu.construct(compute_kwargs={"max_order": 1})
score = bleu.compute(predictions=predictions, references=references)
```

Note that you can seemlessly access both `jury` and `evaluate` metrics through `jury.load_metric`. 

```python
import jury

bleu = jury.load_metric("bleu")
bleu_1 = jury.load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1})
# metrics not available in `jury` but in `evaluate`
wer = jury.load_metric("competition_math") # It falls back to `evaluate` package with a warning
```

### CLI Usage

You can specify predictions file and references file paths and get the resulting scores. Each line should be paired in both files. You can optionally provide reduce function and an export path for results to be written.

    jury eval --predictions /path/to/predictions.txt --references /path/to/references.txt --reduce_fn max --export /path/to/export.txt

You can also provide prediction folders and reference folders to evaluate multiple experiments. In this set up, however, it is required that the prediction and references files you need to evaluate as a pair have the same file name. These common names are paired together for prediction and reference.

    jury eval --predictions /path/to/predictions_folder --references /path/to/references_folder --reduce_fn max --export /path/to/export.txt

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

You can use custom metrics with inheriting `jury.metrics.Metric`, you can see current metrics implemented on Jury from [jury/metrics](https://github.com/obss/jury/tree/master/jury/metrics). Jury falls back to `evaluate` implementation of metrics for the ones that are currently not supported by Jury, you can see the metrics available for `evaluate` on [evaluate/metrics](https://github.com/huggingface/evaluate/tree/master/metrics). 

Jury itself uses `evaluate.Metric` as a base class to drive its own base class as `jury.metrics.Metric`. The interface is similar; however, Jury makes the metrics to take a unified input type by handling the inputs for each metrics, and allows supporting several input types as;

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

Also, you need to install the packages which are available through a git source separately with the following command. 
For the folks who are curious about "why?"; a short explaination is that PYPI does not allow indexing a package which 
are directly dependent on non-pypi packages due to security reasons. The file `requirements-dev.txt` includes packages 
which are currently only available through a git source, or they are PYPI packages with no recent release or 
incompatible with Jury, so that they are added as git sources or pointing to specific commits.

    pip install -r requirements-dev.txt

### Tests

To tests simply run.

    python tests/run_tests.py

### Code Style

To check code style,

    python tests/run_code_style.py check

To format codebase,

    python tests/run_code_style.py format


## <div align="center"> Citation </div>

If you use this package in your work, please cite it as:

    @software{obss2021jury,
      author       = {Cavusoglu, Devrim and Akyon, Fatih Cagatay and Sert, Ulas and Cengiz, Cemil},
      title        = {{Jury: Comprehensive NLP Evaluation toolkit}},
      month        = {feb},
      year         = {2022},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.6108229},
      url          = {https://doi.org/10.5281/zenodo.6108229}
    }

## <div align="center"> License </div>

Licensed under the [MIT](LICENSE) License.
