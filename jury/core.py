from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Mapping, Optional, Union

import datasets

from jury.collator import Collator
from jury.definitions import DEFAULT_METRICS
from jury.metrics import Metric, MetricCollator, load_metric
from jury.utils import replace, set_env


class Jury:
    r"""
    Simple evaluation pipeline for text based metrics. By default it computes BLEU(n),
    METEOR, ROUGE-L metrics.

    Note:

        If ``predictions`` and ``references`` are given as list of strings, the order is recieved
        as prediction & reference pairs and evaluation is done by prioratizing the order.

    Examples:

    .. code-block:: python

        >>> # Question-Generation Evaluation (default BMR)
        >>> predictions = [
            ["the cat is on the mat", "there is playing cat on the mat."],
            ["Look! a wonderful day."]
        ]
        >>> references = [
            ["the cat is playing on the mat.", "The cat plays on the mat."],
            ["Today is a wonderful day", "The weather outside is wonderful."]
        ]
        >>> evaluation = Jury()
        >>> results = evaluation.evaluate(predictions=predictions, references=references)
        >>> print(results)
        {'bleu_1': 0.6111111111111112, ..., 'rougeL': 0.6470588235294118, ...}
    """

    def __init__(
        self,
        metrics: Optional[Union[List[str], List[Metric]]] = None,
        run_concurrent=False,
    ):
        if metrics is None:
            metrics = DEFAULT_METRICS
        else:
            for i, m in enumerate(metrics):
                if isinstance(m, str):
                    metrics = replace(metrics, load_metric(m.lower()), i)

        self.metrics = MetricCollator(metrics)
        self._concurrent = run_concurrent

    def _score_to_dict(self, score, name: str) -> Dict[str, float]:
        if isinstance(score, dict):
            return score

        return {name: score}

    def _compute_single_score(self, inputs) -> Mapping[str, float]:
        metric, predictions, references, reduce_fn = inputs
        if isinstance(metric, Metric):
            predictions, references = Collator(predictions), Collator(references)
            score = metric.compute(predictions=predictions, references=references, reduce_fn=reduce_fn)
        else:
            metric.resulting_name = metric.name
            score = metric.compute(predictions=predictions, references=references)
            score = self._score_to_dict(score, name=metric.name)
        return score

    def _prepare_concurrent_inputs(self, predictions, references, reduce_fn):
        inputs = []
        for metric in self.metrics:
            inputs.append((metric, predictions, references, reduce_fn))
        return inputs

    def evaluate(
        self,
        *,
        predictions: Union[List[str], List[List[str]]] = None,
        references: Union[List[str], List[List[str]]] = None,
        reduce_fn: Optional[Union[str, Callable]] = None,
    ) -> Dict[str, float]:
        if predictions is None or references is None:
            raise TypeError("Both predictions and references have to be passed.")
        if len(predictions) != len(references):
            raise ValueError("Lengths of predictions and references must be equal.")

        metrics = dict()
        metrics["empty_predictions"] = len([1 for p in predictions if not p])
        metrics["total_items"] = len(references)

        if self._concurrent:
            inputs_list = self._prepare_concurrent_inputs(predictions, references, reduce_fn)
            set_env("TOKENIZERS_PARALLELISM", "true")
            with ProcessPoolExecutor() as executor:
                for score in executor.map(self._compute_single_score, inputs_list):
                    metrics.update(score)
        else:
            for metric in self.metrics:
                inputs = (metric, predictions, references, reduce_fn)
                score = self._compute_single_score(inputs)
                metrics.update(score)

        return metrics
