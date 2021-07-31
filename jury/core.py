from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Mapping, Optional, Union

import datasets
import numpy as np
from tqdm import tqdm

from jury.collator import Collator, MetricCollator
from jury.definitions import DEFAULT_METRICS
from jury.tokenizer import BLEUDefaultTokenizer, TokenizerWrapper
from jury.utils import is_reduce_fn, NestedSingleType


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
        metrics: Optional[List[str]] = None,
        run_concurrent=False,
        bleu_tokenizer=None,
    ):
        if metrics is None:
            metrics = DEFAULT_METRICS
        elif NestedSingleType.get_type(metrics) == "list<str>":
            metrics = [metric.lower() for metric in metrics]

        self.metrics = MetricCollator(metrics)
        self.bleu_tokenizer = (
            TokenizerWrapper(bleu_tokenizer) if bleu_tokenizer is not None else TokenizerWrapper(BLEUDefaultTokenizer())
        )
        self._concurrent = run_concurrent

    @staticmethod
    def _compute_metric_for_multiple_items(
        metric: datasets.Metric, predictions: Collator, references: Collator, reduce_fn
    ) -> Dict[str, float]:
        scores = []

        for hyps, refs in tqdm(zip(predictions, references), total=len(references)):
            if len(hyps) == 0:
                # scores.append(0)  # Penalize for not generating any question
                continue

            if "bleu" in metric.metric_name:
                score = [metric.compute(predictions=[hyp], references=[refs], return_dict=False) for hyp in hyps]
            else:
                score = []
                for hyp, ref in zip(hyps, refs):
                    _score = metric.compute(predictions=[hyp], references=[ref], return_dict=False)
                    score.append(_score)
            scores.append(reduce_fn(score))

        return {metric.resulting_name: float(np.mean(scores))}

    def compute_metric(
        self, metric, predictions: Collator, references: Collator, reduce_fn: Callable
    ) -> Dict[str, float]:
        if metric.metric_name == "bleu":
            predictions, references = self.bleu_tokenizer.tokenize(predictions, references)

        if predictions.can_collapse() and references.can_collapse() and "bleu" not in metric.metric_name:
            predictions = Collator(predictions).reshape(-1)
            references = Collator(references).reshape(-1)
            result = metric.compute(predictions=predictions, references=references)
        else:
            result = self._compute_metric_for_multiple_items(metric, predictions=predictions, references=references, reduce_fn=reduce_fn)
        # result = self._postprocess_result(
        #     result,
        #     metric_name=metric_name,
        #     score_name=score_name,
        #     base_name=base_name,
        #     is_datasets_metric=is_datasets_metric,
        # )

        return result

    @staticmethod
    def _postprocess_result(result, metric_name, score_name, base_name, is_datasets_metric: bool):
        if is_datasets_metric:
            if metric_name == "rouge":
                result = {metric_name: result[score_name].mid.fmeasure}
            elif metric_name == "bertscore":
                result = {metric_name: result[score_name][0]}

        if metric_name == "sacrebleu":
            result = {metric_name: result[base_name] / 100}

        return result

    def _compute_single_score(self, inputs) -> Mapping[str, float]:
        metric, predictions, references, reduce_fn = inputs
        score = self.compute_metric(metric=metric, predictions=predictions, references=references, reduce_fn=reduce_fn)
        return score

    def _prepare_concurrent_inputs(self, predictions, references, reduce_fn):
        inputs = []
        for metric in self.metrics:
            inputs.append([metric, predictions, references, reduce_fn])
        return inputs

    def evaluate(
        self,
        predictions: Union[str, List[str], List[List[str]], List[Dict]],
        references: Union[str, List[str], List[List[str]], List[Dict]],
        reduce_fn: Union[str, Callable] = "max",
    ) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Lengths of predictions and references must be equal.")

        reduce_fn = getattr(np, reduce_fn) if isinstance(reduce_fn, str) else reduce_fn
        if not is_reduce_fn(reduce_fn):
            raise ValueError("'reduce_fn' must be an aggregation function.")
        predictions = Collator(predictions)
        references = Collator(references)
        metrics = {}

        empty_predictions = 0
        for preds in tqdm(predictions):
            if not preds:  # Check if empty
                empty_predictions += 1
        metrics["empty_predictions"] = empty_predictions
        metrics["total_items"] = len(references)

        inputs_list = self._prepare_concurrent_inputs(predictions, references, reduce_fn)

        if self._concurrent:
            with ProcessPoolExecutor() as executor:
                for score in executor.map(self._compute_single_score, inputs_list):
                    metrics.update(score)
        else:
            for inputs in inputs_list:
                score = self._compute_single_score(inputs)
                metrics.update(score)

        return metrics
