from jury.metrics._core import MetricAlias
from jury.metrics.wer.wer_for_language_generation import WERForLanguageGeneration

__main_class__ = "WER"


class WER(MetricAlias):
    _SUBCLASS = WERForLanguageGeneration
