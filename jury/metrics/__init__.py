from jury.metrics._core import (
    AutoMetric,
    EvaluationInstance,
    LanguageGenerationInstance,
    Metric,
    MetricForLanguageGeneration,
    MetricForSequenceClassification,
    MetricForSequenceLabeling,
    MetricForTask,
    SequenceClassificationInstance,
    SequenceLabelingInstance,
    list_metrics,
    load_metric,
)
from jury.metrics.accuracy import Accuracy
from jury.metrics.bartscore import Bartscore
from jury.metrics.bertscore import Bertscore
from jury.metrics.bleu import Bleu
from jury.metrics.bleurt import Bleurt
from jury.metrics.cer import CER
from jury.metrics.comet import Comet
from jury.metrics.f1 import F1
from jury.metrics.meteor import Meteor
from jury.metrics.precision import Precision
from jury.metrics.prism import Prism
from jury.metrics.recall import Recall
from jury.metrics.rouge import Rouge
from jury.metrics.sacrebleu import Sacrebleu
from jury.metrics.squad import Squad
from jury.metrics.wer import WER

if __name__ == "__main__":
    from jury.metrics.comet.comet_for_language_generation import CometForCrossLingualEvaluation

    source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
    hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
    reference = ["They were able to control the fire.", "Schools and kindergartens opened"]

    comet = CometForCrossLingualEvaluation()
    res = comet.compute(sources=source, predictions=hypothesis, references=reference)
    print(res)
