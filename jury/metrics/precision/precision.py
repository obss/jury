from jury.metrics._core import TaskMapper
from jury.metrics._core.utils import TaskNotAvailable
from jury.metrics.precision.precision_for_language_generation import PrecisionForLanguageGeneration
from jury.metrics.precision.precision_for_sequence_classification import PrecisionForSequenceClassification

__main_class__ = "Precision"


class Precision(TaskMapper):
    _TASKS = {
        "language-generation": PrecisionForLanguageGeneration,
        "sequence-classification": PrecisionForSequenceClassification,
    }
