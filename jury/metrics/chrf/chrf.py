from jury.metrics._core import MetricAlias
from jury.metrics.chrf.chrf_for_language_generation import CHRFForLanguageGeneration

__main_class__ = "CHRF"


class CHRF(MetricAlias):
    _SUBCLASS = CHRFForLanguageGeneration
