from jury.metrics import Metric
from jury.metrics.bleu import Bleu
from jury.metrics.meteor import Meteor
from jury.metrics.rouge import Rouge

_METRIC_DEFINITIONS = {
    "squad": {"metric_name": "squad", "score_name": None},
    "bleu_1": {"metric_name": "bleu", "max_order": 1, "score_name": "bleu"},
    "bleu_2": {"metric_name": "bleu", "max_order": 2, "score_name": "bleu"},
    "bleu_3": {"metric_name": "bleu", "max_order": 3, "score_name": "bleu"},
    "bleu_4": {"metric_name": "bleu", "max_order": 4, "score_name": "bleu"},
    "meteor": {"metric_name": "meteor", "score_name": "meteor"},
    "rouge": {"metric_name": "rouge", "rouge_types": ["rougeL"], "score_name": "rougeL"},
    "sacrebleu": {"metric_name": "sacrebleu", "score_name": "score"},
    "bertscore": {"metric_name": "bertscore", "lang": "en", "score_name": "f1"},
    "bleurt": {"metric_name": "bleurt", "score_name": "scores"},
}


DEFAULT_METRICS = [
    # "squad": {"metric_name": "squad", "score_name": None},
    Bleu(resulting_name="bleu_1", params={"max_order": 1}),
    Bleu(resulting_name="bleu_2", params={"max_order": 2}),
    Bleu(resulting_name="bleu_3", params={"max_order": 3}),
    Bleu(resulting_name="bleu_4", params={"max_order": 4}),
    Meteor(),
    Rouge(),
    # "sacrebleu": {"metric_name": "sacrebleu", "score_name": "score"},
    # "bertscore": {"metric_name": "bertscore", "lang": "en", "score_name": "f1"},
    # "bleurt": {"metric_name": "bleurt", "score_name": "scores"},
]


