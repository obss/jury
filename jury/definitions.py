from jury.metrics import load_metric

DEFAULT_METRICS = [
    load_metric("bleu", compute_kwargs={"max_order": 1}),
    load_metric("bleu", compute_kwargs={"max_order": 2}),
    load_metric("bleu", compute_kwargs={"max_order": 3}),
    load_metric("bleu", compute_kwargs={"max_order": 4}),
    load_metric("meteor"),
    load_metric("rouge"),
]
