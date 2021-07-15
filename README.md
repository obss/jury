# Jury

Simple tool/toolkit for evaluating NLG (Natural Language Generation) offering various automated metrics. Jury offers a smooth and easy-to-use interface. It uses huggingface/datasets package for underlying metric computation, and hence adding custom metric is easy as adopting `datasets.Metric`.

## Installation

Through pip,

    pip install jury

or build from source,

    git clone https://github.com/obss/jury.git
    cd jury
    python setup.py install

## Basic Usage

### API Usage

It is only two lines of code to evaluate generated outputs.

    from jury import Jury
    
    jury = Jury()
    scores = jury.evaluate(predictions, references)


Specify metrics you want to use on instantiation.

    jury = Jury(metrics=["bleu", "meteor"])
    scores = jury.evaluate(predictions, references)

### CLI Usage

Coming soon...


## License

Licensed under the [MIT](LICENSE) License.
