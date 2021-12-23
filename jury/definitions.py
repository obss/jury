DEFAULT_METRICS = [
    {"path": "bleu", "compute_kwargs": {"max_order": 1}},
    {"path": "bleu", "compute_kwargs": {"max_order": 2}},
    {"path": "bleu", "compute_kwargs": {"max_order": 3}},
    {"path": "bleu", "compute_kwargs": {"max_order": 4}},
    {"path": "meteor"},
    {"path": "rouge"},
]
