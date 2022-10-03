from jury.metrics._core import MetricAlias
from jury.metrics.seqeval.seqeval_for_sequence_labeling import SeqevalForSequnceLabeling

__main_class__ = "Seqeval"


class Seqeval(MetricAlias):
    _SUBCLASS = SeqevalForSequnceLabeling
