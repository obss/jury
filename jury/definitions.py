DEFAULT_METRICS = [
    {"metric_name": "bleu", "compute_kwargs": {"max_order": 1}},
    {"metric_name": "bleu", "compute_kwargs": {"max_order": 2}},
    {"metric_name": "bleu", "compute_kwargs": {"max_order": 3}},
    {"metric_name": "bleu", "compute_kwargs": {"max_order": 4}},
    {"metric_name": "meteor"},
    {"metric_name": "rouge"},
]
