from jury.metrics.bleu import BleuForLanguageGeneration as Bleu
from jury.metrics.meteor import MeteorForLanguageGeneration as Meteor
from jury.metrics.rouge import RougeForLanguageGeneration as Rouge

DEFAULT_METRICS = [
    Bleu(compute_kwargs={"max_order": 1}),
    Bleu(compute_kwargs={"max_order": 2}),
    Bleu(compute_kwargs={"max_order": 3}),
    Bleu(compute_kwargs={"max_order": 4}),
    Meteor(),
    Rouge(),
]
