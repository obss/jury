# coding=utf-8
# Copyright 2020 Open Business Software Solutions, The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ROUGE metric. The part of this file is adapted from HuggingFace's
datasets package implementation of ROUGE metric. See
https://github.com/huggingface/datasets/blob/master/metrics/rouge/rouge.py
"""

from typing import Callable, Dict, List, Optional, Union

import datasets
import pandas as pd
from rouge_score import rouge_scorer, scoring

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration

_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""

_DESCRIPTION = """\
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for
evaluating automatic summarization and machine translation software in natural language processing.
The metrics compare an automatically produced summary or translation against a reference or a set of references 
(human-produced) summary or translation.

Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.

This metrics is a wrapper around Google Research reimplementation of ROUGE:
https://github.com/google-research/google-research/tree/master/rouge
"""

_KWARGS_DESCRIPTION = """
Calculates average rouge scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    rouge_types: A list of rouge types to calculate.
        Valid names:
        `"rouge{n}"` (e.g. `"rouge1"`, `"rouge2"`) where: {n} is the n-gram based scoring,
        `"rougeL"`: Longest common subsequence based scoring.
        `"rougeLSum"`: rougeLsum splits text using `"\n"`.
        See details in https://github.com/huggingface/datasets/issues/617
    use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.
    use_agregator: Return aggregates if this is set to True
Returns:
    rouge1: rouge_1 (precision, recall, f1),
    rouge2: rouge_2 (precision, recall, f1),
    rougeL: rouge_l (precision, recall, f1),
    rougeLsum: rouge_lsum (precision, recall, f1)
Examples:

    >>> rouge = jury.load_metric("rouge")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(results)
    {'rouge': {'rouge1': 0.7783882783882783, 'rouge2': 0.5925324675324675, 
        'rougeL': 0.7426739926739926, 'rougeLsum': 0.7426739926739926}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RougeForLanguageGeneration(MetricForLanguageGeneration):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    @staticmethod
    def _get_aggregator(use_aggregator: bool) -> Union[List, scoring.BootstrapAggregator]:
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            aggregator = []
        return aggregator

    @staticmethod
    def _add_score(
        aggregator: Union[List, scoring.BootstrapAggregator],
        score: Union[Dict[str, float], Dict[str, scoring.BootstrapAggregator]],
    ) -> Union[List, scoring.BootstrapAggregator]:
        if isinstance(aggregator, scoring.BootstrapAggregator):
            aggregator.add_scores(score)
        else:
            aggregator.append(score)
        return aggregator

    @staticmethod
    def _aggregate(aggregator) -> Union[Dict[str, float], Dict[str, scoring.AggregateScore]]:
        if isinstance(aggregator, scoring.BootstrapAggregator):
            result = aggregator.aggregate()
        else:
            result = {}
            for key in aggregator[0]:
                result[key] = list(score[key] for score in aggregator)
        return result

    @staticmethod
    def _normalize_score_list(
        score_list: List[Dict[str, scoring.BootstrapAggregator]], metric_to_select: str
    ) -> List[Dict[str, scoring.BootstrapAggregator]]:
        for score_dict in score_list:
            for metric, score in score_dict.items():
                score_dict[metric] = getattr(score, metric_to_select)
        return score_list

    @staticmethod
    def _reduce_dict(score_list: List[Dict[str, scoring.BootstrapAggregator]], reduce_fn: callable) -> Dict[str, float]:
        return pd.DataFrame(score_list).apply(reduce_fn, axis=0).to_dict()

    @staticmethod
    def _select_mid_from_aggregation(aggregated_scores: Dict[str, scoring.AggregateScore]) -> Dict[str, float]:
        return {metric: score.mid for metric, score in aggregated_scores.items()}

    def evaluate(
        self,
        *,
        predictions: Collator = None,
        references: Collator = None,
        reduce_fn: Callable = None,
        rouge_types: List[str] = None,
        use_aggregator: bool = True,
        use_stemmer: bool = False,
        metric_to_select: Optional[str] = "fmeasure",
        **kwargs
    ):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)

        if predictions.can_collapse() and references.can_collapse():
            aggregator = self._get_aggregator(use_aggregator)
            predictions = predictions.collapse()
            references = references.collapse()
            aggregator = self._compute_single_pred_single_ref(
                predictions, references, reduce_fn, scorer, aggregator, metric_to_select
            )
        elif predictions.can_collapse() and not references.can_collapse():
            # Force to use BootstrapAggregator
            aggregator = self._get_aggregator(use_aggregator=True)
            predictions = predictions.collapse()
            aggregator = self._compute_single_pred_multi_ref(
                predictions, references, reduce_fn, scorer, aggregator, metric_to_select
            )
        else:
            # Force to use BootstrapAggregator
            aggregator = self._get_aggregator(use_aggregator=True)
            aggregator = self._compute_multi_pred_multi_ref(
                predictions, references, reduce_fn, scorer, aggregator, metric_to_select
            )

        result = self._aggregate(aggregator)

        if metric_to_select and use_aggregator:
            result = self._select_mid_from_aggregation(result)
        elif metric_to_select and not use_aggregator:
            result = {k: v[0] for k, v in result.items()}

        return result

    def _compute_single_pred_single_ref(
        self, predictions, references, reduce_fn: Callable = None, scorer=None, aggregator=None, metric_to_select=None
    ):
        for ref, pred in zip(references, predictions):
            score = scorer.score(target=ref, prediction=pred)
            if metric_to_select is not None:
                score = self._normalize_score_list(score_list=[score], metric_to_select=metric_to_select)[0]
            aggregator = self._add_score(aggregator, score)
        return aggregator

    def _compute_single_pred_multi_ref(
        self, predictions, references, reduce_fn: Callable = None, scorer=None, aggregator=None, metric_to_select=None
    ):
        for pred, refs in zip(predictions, references):
            pred_scores = [scorer.score(target=ref, prediction=pred) for ref in refs]
            pred_scores = self._normalize_score_list(pred_scores, metric_to_select)
            score = self._reduce_dict(pred_scores, reduce_fn)
            aggregator = self._add_score(aggregator, score)
        return aggregator

    def _compute_multi_pred_multi_ref(
        self, predictions, references, reduce_fn: Callable = None, scorer=None, aggregator=None, metric_to_select=None
    ):
        for preds, refs in zip(predictions, references):
            multi_aggregator = self._get_aggregator(use_aggregator=True)
            for pred in preds:
                pred_score = self._normalize_score_list(
                    [scorer.score(target=ref, prediction=pred) for ref in refs], metric_to_select
                )
                pred_score = self._reduce_dict(pred_score, reduce_fn)
                multi_aggregator = self._add_score(multi_aggregator, pred_score)
            score = self._select_mid_from_aggregation(self._aggregate(multi_aggregator))
            aggregator = self._add_score(aggregator, score)
        return aggregator
