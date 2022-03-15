from jury.metrics._core import MetricAlias
from jury.metrics.ter.ter_for_language_generation import TERForLanguageGeneration

__main_class__ = "TER"


class TER(MetricAlias):
    _SUBCLASS = TERForLanguageGeneration
