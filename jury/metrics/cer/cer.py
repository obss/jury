from jury.metrics._core import MetricAlias
from jury.metrics.cer.cer_for_language_generation import CERForLanguageGeneration

__main_class__ = "CER"


class CER(MetricAlias):
    _SUBCLASS = CERForLanguageGeneration
