from jury.metrics._core import TaskMapper
from jury.metrics.f1.f1_for_language_generation import F1ForLanguageGeneration
from jury.metrics.f1.f1_for_sequence_classification import F1ForSequenceClassification

__main_class__ = "F1"


class F1(TaskMapper):
    _TASKS = {"language-generation": F1ForLanguageGeneration, "sequence-classification": F1ForSequenceClassification}
