from jury.metrics._core import MetricAlias
from jury.metrics.rouge.rouge_for_language_generation import RougeForLanguageGeneration

__main_class__ = "Rouge"


class Rouge(MetricAlias):
    _SUBCLASS = RougeForLanguageGeneration
