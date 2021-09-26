from jury.metrics import Bleu, Meteor, Rouge

DEFAULT_METRICS = [
    Bleu(params={"max_order": 1}),
    Bleu(params={"max_order": 2}),
    Bleu(params={"max_order": 3}),
    Bleu(params={"max_order": 4}),
    Meteor(),
    Rouge(),
]
