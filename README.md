<div align="center">
<h1>
Jury
</h1>
</div>

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

    # Microsoft translator translition for "Yurtta sulh, cihanda sulh." (16.07.2021)
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
    cd pigeons
    python setup.py develop
    pip install -r requirements-dev.txt

### Tests

To tests simply run.

    bash tests/run_tests.sh

### Code Style

To check code style,

    bash tests/run_code_style.sh check

To format codebase,

    bash tests/run_code_style.sh format


## <div align="center"> License </div>

Licensed under the [MIT](LICENSE) License.
