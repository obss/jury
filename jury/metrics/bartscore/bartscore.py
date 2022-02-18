from jury.metrics._core import MetricAlias
from jury.metrics.bartscore.bartscore_for_language_generation import BartscoreForLanguageGeneration

__main_class__ = "Bartscore"


class Bartscore(MetricAlias):
    _SUBCLASS = BartscoreForLanguageGeneration
