from jury.metrics.bleu import Bleu
from jury.metrics.meteor import Meteor
from jury.metrics.rouge import Rouge
from jury.metrics.sacrebleu import Sacrebleu

DEFAULT_METRICS = [
    Bleu(params={"max_order": 1}),
    Bleu(params={"max_order": 2}),
    Bleu(params={"max_order": 3}),
    Bleu(params={"max_order": 4}),
    Meteor(),
    Rouge(),
    Sacrebleu(),
]
