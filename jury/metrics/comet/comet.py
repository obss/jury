from jury.metrics._core import MetricAlias
from jury.metrics.comet.comet_for_language_generation import CometForCrossLingualEvaluation

__main_class__ = "Comet"


class Comet(MetricAlias):
    _SUBCLASS = CometForCrossLingualEvaluation
