from jury.metrics._core import MetricAlias
from jury.metrics.prism.prism_for_language_generation import PrismForLanguageGeneration

__main_class__ = "Prism"


class Prism(MetricAlias):
    _SUBCLASS = PrismForLanguageGeneration
