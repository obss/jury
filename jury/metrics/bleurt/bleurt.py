from jury.metrics._core import MetricAlias
from jury.metrics.bleurt.bleurt_for_language_generation import BleurtForLanguageGeneration

__main_class__ = "Bleurt"


class Bleurt(MetricAlias):
    _SUBCLASS = BleurtForLanguageGeneration
