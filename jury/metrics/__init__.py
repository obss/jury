from jury.metrics._core import AutoMetric, Metric, load_metric
from jury.metrics.accuracy import AccuracyForLanguageGeneration
from jury.metrics.bertscore import Bertscore
from jury.metrics.bleu import Bleu
from jury.metrics.f1 import F1ForLanguageGeneration
from jury.metrics.meteor import Meteor
from jury.metrics.precision import Precision
from jury.metrics.recall import RecallForLanguageGeneration
from jury.metrics.rouge import Rouge
from jury.metrics.sacrebleu import Sacrebleu
from jury.metrics.squad import Squad

# TODO: Implement factory metrics and separate metrics into sub-groups for different tasks.

__all__ = [
    "AccuracyForLanguageGeneration",
    "AutoMetric",
    "Bertscore",
    "Bleu",
    "F1ForLanguageGeneration",
    "Meteor",
    "Precision",
    "RecallForLanguageGeneration",
    "Rouge",
    "Sacrebleu",
    "Squad",
    "load_metric",
]
