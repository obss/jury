from jury.metrics._core import MetricAlias
from jury.metrics.sacrebleu.sacrebleu_for_language_generation import SacrebleuForLanguageGeneration

__main_class__ = "Sacrebleu"


class Sacrebleu(MetricAlias):
    _SUBCLASS = SacrebleuForLanguageGeneration
