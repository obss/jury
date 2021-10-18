from jury.metrics._core import MetricAlias
from jury.metrics.bertscore.bertscore_for_language_generation import BertscoreForLanguageGeneration

__main_class__ = "Bertscore"


class Bertscore(MetricAlias):
    _SUBCLASS = BertscoreForLanguageGeneration
