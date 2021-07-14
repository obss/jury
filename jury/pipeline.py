from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Dict, Union, Tuple

import datasets
import numpy as np
from tqdm import tqdm

from jury.definitions import METRIC_PARAMS
from jury.utils import remove_punctuations


class Jury:
    r"""
    Simple evaluation pipeline for text based metrics. By default it computes BLEU(n),
    METEOR, ROUGE-L and SacreBLEU metrics.

    Note:

                    If ``predictions`` and ``references`` are given as list of strings, the order is recieved
                    as prediction & reference pairs and evaluation is done by prioratizing the order.

    Examples:

    .. code-block:: python

                    >>> # Question-Generation Evaluation (default BMR)
                    >>>	predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
                    >>> references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
                    >>> evaluation = Jury()
                    >>> results = evaluation.evaluate(predictions=predictions, references=references)
                    >>> print(results)
                    {'bleu_1': 0.6111111111111112, ..., 'rougeL': 0.6470588235294118, ...}
    """

    def __init__(self, metrics: Optional[List[str]] = None, preload_metrics=False):
        if metrics is None:
            metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge", "sacrebleu", "bertscore"]
        else:
            metrics = [metric.lower() for metric in metrics]

        self.metrics = metrics
        self.preloaded_metrics = None
        self.preload_metrics = preload_metrics

        if preload_metrics:
            self._preload_metrics()

    @staticmethod
    def _get_params_from_metric(metric: str):
        try:
            params = METRIC_PARAMS[metric]
            params["base_name"] = metric
        except KeyError:
            raise ValueError(f"Unknown metric {metric}.")

        return params

    def _preload_metrics(self) -> None:
        preload_metrics = {}
        for metric in self.metrics:
            params = self._get_params_from_metric(metric)
            preload_metrics[metric] = datasets.load_metric(params["metric_name"])
        setattr(self, "preloaded_metrics", preload_metrics)

    def load_metric(self, name) -> datasets.Metric:
        if self.preload_metrics and name in self.preloaded_metrics:
            return self.preloaded_metrics[name]
        return datasets.load_metric(name)

    @staticmethod
    def _compute_metric_for_multiple_candidates(
        metric: datasets.Metric, predictions: List[List[str]], references: List[str], reduce="mean", **kwargs
    ) -> Dict[str, float]:
        scores = []
        reduce_fn = getattr(np, reduce)
        score_name = kwargs.pop("score_name")
        base_name = kwargs.pop("base_name")

        for hyps, ref in tqdm(zip(predictions, references), total=len(references)):
            if len(hyps) == 0:
                # scores.append(0)  # Penalize for not generating any question
                continue
            score = [metric.compute(predictions=[hyp], references=[ref], **kwargs)[score_name] for hyp in hyps]
            scores.append(reduce_fn(score))

        return {base_name: float(np.mean(scores))}

    @staticmethod
    def compute_individual_metric(metric, prediction: str, reference: str, **kwargs) -> Dict:
        return metric.compute(predictions=[prediction], references=[[reference]], **kwargs)

    def compute_metric(
        self, metric_name: str, predictions: List[str], references: List[str], **kwargs
    ) -> Dict[str, float]:
        def _tokenize_for_bleu(
            predictions: Union[List[str], List[List[str]]], references: List[str]
        ) -> Tuple[List[List[str]], List[List[List[str]]]]:
            if isinstance(predictions[0], str):
                predictions = [remove_punctuations(pred).split() for pred in predictions]
            else:
                _predictions = []
                for preds in predictions:
                    _predictions.append([remove_punctuations(pred).split() for pred in preds])
                predictions = _predictions
            references = [[remove_punctuations(ref).split()] for ref in references]
            return predictions, references

        assert len(predictions) == len(references), "Currently supporting 1 reference per candidate"

        if metric_name == "bleu":
            predictions, references = _tokenize_for_bleu(predictions, references)

        metric = self.load_metric(metric_name)
        if metric_name == "sacrebleu" and not isinstance(references[0], list):
            references = [[ref] for ref in references]

        if isinstance(predictions[0], list):
            result = self._compute_metric_for_multiple_candidates(
                metric, predictions=predictions, references=references, **kwargs
            )
        else:
            result = metric.compute(predictions=predictions, references=references, **kwargs)

            if metric_name == "rouge":
                result = {"rougeL": result["rougeL"].mid.fmeasure}
        if metric_name == "sacrebleu":
            result = {metric_name: result[metric_name] / 100}
        return result

    def _prepare_concurrent_inputs(self, predictions, references):
        inputs = []
        for metric in self.metrics:
            inputs.append([metric, predictions, references])
        return inputs

    def _compute_single_score(self, inputs) -> Dict[str, float]:
        metric_name, predictions, references = inputs
        params = self._get_params_from_metric(metric_name)
        score = self.compute_metric(predictions=predictions, references=references, **params)
        return score

    def evaluate(
        self, predictions: Union[List[str], List[List[str]], List[Dict]], references: Union[List[str], List[Dict]]
    ) -> Dict[str, float]:
        metrics = {}

        empty_questions = 0
        for preds in tqdm(predictions):
            if not preds:  # Check if empty
                empty_questions += 1
        metrics["empty_questions"] = empty_questions
        metrics["total_questions"] = len(references)

        inputs_list = self._prepare_concurrent_inputs(predictions, references)

        with ProcessPoolExecutor() as executor:
            for score in executor.map(self._compute_single_score, inputs_list):
                metrics.update(score)

        return metrics
