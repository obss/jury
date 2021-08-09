from jury.metrics.bleu import BLEU
from jury.metrics.meteor import Meteor
from jury.metrics.rouge import Rouge
from jury.metrics.sacrebleu import SacreBLEU

DEFAULT_METRICS = [
    BLEU(resulting_name="bleu_1", params={"max_order": 1}),
    BLEU(resulting_name="bleu_2", params={"max_order": 2}),
    BLEU(resulting_name="bleu_3", params={"max_order": 3}),
    BLEU(resulting_name="bleu_4", params={"max_order": 4}),
    Meteor(),
    Rouge(),
    SacreBLEU(),
]
