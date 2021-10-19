from jury.metrics._core import TaskMapper
from jury.metrics.recall.recall_for_language_generation import RecallForLanguageGeneration
from jury.metrics.recall.recall_for_sequence_classification import RecallForSequenceClassification

__main_class__ = "Recall"


class Recall(TaskMapper):
    _TASKS = {
        "language-generation": RecallForLanguageGeneration,
        "sequence-classification": RecallForSequenceClassification,
    }
