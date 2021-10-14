from jury.metrics import Bleu, Meteor, Rouge

DEFAULT_METRICS = [
    Bleu(compute_kwargs={"max_order": 1}),
    Bleu(compute_kwargs={"max_order": 2}),
    Bleu(compute_kwargs={"max_order": 3}),
    Bleu(compute_kwargs={"max_order": 4}),
    Meteor(),
    Rouge(),
]
