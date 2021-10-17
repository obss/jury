from jury.metrics import AutoMetric

DEFAULT_METRICS = [
    AutoMetric.from_params("bleu", compute_kwargs={"max_order": 1}),
    AutoMetric.from_params("bleu", compute_kwargs={"max_order": 2}),
    AutoMetric.from_params("bleu", compute_kwargs={"max_order": 3}),
    AutoMetric.from_params("bleu", compute_kwargs={"max_order": 4}),
    AutoMetric.from_params("meteor"),
    AutoMetric.from_params("rouge"),
]
