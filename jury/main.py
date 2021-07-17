from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Dict, Union, Tuple

import datasets
import numpy as np
from tqdm import tqdm

from jury.core import InputList
from jury.definitions import METRIC_DEFINITIONS
from jury.tokenizer import BLEUDefaultTokenizer, TokenizerWrapper


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
    # TODO: if sequence is collapsable, compute with datasets.Metric.compute, else fallback to own computation.
    _DEFAULT_METRICS = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge"]

    def __init__(self, metrics: Optional[List[str]] = None, preload_metrics: bool = False, bleu_tokenizer=None):
        self.bleu_tokenizer = TokenizerWrapper(bleu_tokenizer) if bleu_tokenizer is not None \
                                else TokenizerWrapper(BLEUDefaultTokenizer())
        if metrics is None:
            metrics = self._DEFAULT_METRICS
        else:
            metrics = [metric.lower() for metric in metrics]

        self.metrics = metrics
        self._preloaded_metrics = None
        self.preload_metrics = preload_metrics

        if preload_metrics:
            self._preload_metrics()

    @staticmethod
    def _get_metric_definition(metric_name: str):
        try:
            definition = METRIC_DEFINITIONS[metric_name]
            definition["base_name"] = metric_name
        except KeyError:
            raise ValueError(f"Unknown metric {metric_name}.")

        return definition

    def _preload_metrics(self) -> None:
        preloaded_metrics = {}
        for metric in self.metrics:
            params = self._get_metric_definition(metric)
            preloaded_metrics[metric] = datasets.load_metric(params["metric_name"])
        setattr(self, "preloaded_metrics", preloaded_metrics)

    def load_metric(self, name) -> datasets.Metric:
        if self.preload_metrics and name in self._preloaded_metrics:
            return self._preloaded_metrics[name]
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
    def compute_single_item(metric, prediction: str, reference: str, **kwargs) -> Dict:
        return metric.compute(predictions=[prediction], references=[[reference]], **kwargs)

    def compute_metric(
        self, metric_name: str, predictions: InputList, references: InputList, **kwargs
    ) -> Dict[str, float]:
        metric = self.load_metric(metric_name)
        if isinstance(predictions[0], list):
            compute_fn = self._compute_metric_for_multiple_candidates
            kwargs["metric"] = metric
        else:
            compute_fn = metric.compute
            kwargs.pop("score_name")
            kwargs.pop("base_name")

        if metric_name == "bleu":
            predictions, references = self.bleu_tokenizer.tokenize(predictions, references)
        elif metric_name == "sacrebleu" and not isinstance(references[0], list):
            references = [[ref] for ref in references]

        result = compute_fn(predictions=predictions, references=references, **kwargs)

        if metric_name == "rouge":
            result = {"rougeL": result["rougeL"].mid.fmeasure}
        elif metric_name == "sacrebleu":
            result = {metric_name: result["score"] / 100}
        return result

    def _compute_single_score(self, inputs) -> Dict[str, float]:
        metric_name, predictions, references = inputs
        params = self._get_metric_definition(metric_name)
        score = self.compute_metric(predictions=predictions, references=references, **params)
        return score

    def _prepare_concurrent_inputs(self, predictions, references):
        inputs = []
        for metric in self.metrics:
            inputs.append([metric, predictions, references])
        return inputs

    def evaluate(
            self,
            predictions: Union[str, List[str], List[List[str]], List[Dict]],
            references: Union[str, List[str], List[List[str]], List[Dict]]
    ) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Lengths of predictions and references must be equal.")

        predictions = InputList(predictions)
        references = InputList(references)
        metrics = {}

        empty_predictions = 0
        for preds in tqdm(predictions):
            if not preds:  # Check if empty
                empty_predictions += 1
        metrics["empty_predictions"] = empty_predictions
        metrics["total_items"] = len(references)

        inputs_list = self._prepare_concurrent_inputs(predictions, references)

        for inputs in inputs_list:
            score = self._compute_single_score(inputs)
            metrics.update(score)

        # with ProcessPoolExecutor() as executor:
        #     for score in executor.map(self._compute_single_score, inputs_list):
        #         metrics.update(score)

        return metrics


if __name__ == "__main__":
    import json

    jury = Jury()

    preds = ["abc def"]
    refs = ["abc def"]
    scores = jury.evaluate(preds, refs)
    print(json.dumps(scores, indent=4))
