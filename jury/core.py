from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from jury.collator import Collator
from jury.definitions import DEFAULT_METRICS
from jury.metrics import Metric, load_metric
from jury.utils import replace, set_env

MetricParam = Union[str, Metric, Dict[str, Any]]


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
        >>> results = evaluation(predictions=predictions, references=references)
        >>> print(results)
        {'bleu_1': 0.6111111111111112, ..., 'rougeL': 0.6470588235294118, ...}
    """

    def __init__(
        self,
        metrics: Optional[Union[MetricParam, List[MetricParam]]] = None,
        run_concurrent=False,
    ):
        self.metrics = self._load_metrics(metrics)
        self._concurrent = run_concurrent

    def __call__(
        self,
        *,
        predictions: Union[List[str], List[List[str]]] = None,
        references: Union[List[str], List[List[str]]] = None,
        reduce_fn: Optional[Union[str, Callable]] = None,
    ) -> Dict[str, float]:
        """Restricts positional arguments to prevent potential inconsistency between predictions and references."""
        if predictions is None or references is None:
            raise TypeError("Both predictions and references have to be passed.")
        if len(predictions) != len(references):
            raise ValueError("Lengths of predictions and references must be equal.")

        scores = dict()
        scores["empty_predictions"] = len([1 for p in predictions if not p])
        scores["total_items"] = len(references)

        if self._concurrent:
            inputs_list = self._prepare_concurrent_inputs(predictions, references, reduce_fn)
            set_env("TOKENIZERS_PARALLELISM", "true")
            with ProcessPoolExecutor() as executor:
                for score in executor.map(self._compute_single_score, inputs_list):
                    scores.update(score)
        else:
            for metric in self.metrics:
                inputs = (metric, predictions, references, reduce_fn)
                score = self._compute_single_score(inputs)
                scores.update(score)

        return scores

    def _load_single_metric(self, metric: Union[str, Metric]) -> List[Metric]:
        if isinstance(metric, str):
            metric = load_metric(metric)
        return [metric]

    def _load_multiple_metrics(self, metrics: Union[List[str], List[Dict[str, Any]], List[Metric]]) -> List[Metric]:
        for i, metric_param in enumerate(metrics):
            if isinstance(metric_param, str):
                metric_name = metric_param
                metrics = replace(metrics, load_metric(metric_name.lower()), i)
            elif isinstance(metric_param, dict):
                metric_name = metric_param.pop("metric_name")  # must be given
                resulting_name = metric_param.pop("resulting_name") if "resulting_name" in metric_param else None
                params = metric_param
                metrics = replace(
                    metrics, load_metric(metric_name=metric_name, resulting_name=resulting_name, params=params), i
                )
            elif isinstance(metric_param, Metric):
                continue
        return metrics

    def _load_metrics(self, metrics: Union[MetricParam, List[MetricParam]]) -> List[Metric]:
        if metrics is None:
            metrics = DEFAULT_METRICS
        elif isinstance(metrics, (str, Metric)):
            metrics = self._load_single_metric(metrics)
        elif isinstance(metrics, list):
            metrics = self._load_multiple_metrics(metrics)
        else:
            raise ValueError(f"Unknown input type {type(metrics)}")

        return metrics

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

    def add_metric(self, metric_name: str, resulting_name: str = None, params: Dict = None) -> None:
        metric = load_metric(metric_name, resulting_name=resulting_name, params=params)
        self.metrics.append(metric)

    def remove_metric(self, resulting_name: str, error: bool = True) -> None:
        for i, metric in enumerate(self.metrics):
            if metric.resulting_name == resulting_name:
                self.metrics.pop(i)
                return
        if error:
            # raise an error if resulting_metric is not found
            raise ValueError(f"Metric with resulting name {resulting_name} does not exists.")

    def evaluate(
        self,
        *,
        predictions: Union[List[str], List[List[str]]] = None,
        references: Union[List[str], List[List[str]]] = None,
        reduce_fn: Optional[Union[str, Callable]] = None,
    ) -> Dict[str, float]:
        """Returns __call__() method. For backward compatibility."""
        return self.__call__(predictions=predictions, references=references, reduce_fn=reduce_fn)
