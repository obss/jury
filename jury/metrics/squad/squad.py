from jury.metrics._core import MetricAlias
from jury.metrics.squad.squad_for_language_generation import SquadForLanguageGeneration

__main_class__ = "Squad"


class Squad(MetricAlias):
    _SUBCLASS = SquadForLanguageGeneration
