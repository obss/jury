from jury.metrics._core import AutoMetric, load_metric
from jury.metrics.accuracy import Accuracy
from jury.metrics.bertscore import Bertscore
from jury.metrics.bleu import Bleu
from jury.metrics.f1 import F1
from jury.metrics.meteor import Meteor
from jury.metrics.precision import Precision
from jury.metrics.recall import Recall
from jury.metrics.rouge import Rouge
from jury.metrics.sacrebleu import Sacrebleu
from jury.metrics.squad import Squad

# TODO: Implement factory metrics and separate metrics into sub-groups for different tasks.

__all__ = [
    "Accuracy",
    "AutoMetric",
    "Bertscore",
    "Bleu",
    "F1",
    "Meteor",
    "Precision",
    "Recall",
    "Rouge",
    "Sacrebleu",
    "Squad",
    "load_metric",
]
