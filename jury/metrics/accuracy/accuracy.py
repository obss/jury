from jury.metrics._core import TaskMapper
from jury.metrics.accuracy.accuracy_for_language_generation import AccuracyForLanguageGeneration
from jury.metrics.accuracy.accuracy_for_sequence_classification import AccuracyForSequenceClassification

__main_class__ = "Accuracy"


class Accuracy(TaskMapper):
    _TASKS = {
        "language-generation": AccuracyForLanguageGeneration,
        "sequence-classification": AccuracyForSequenceClassification,
    }
