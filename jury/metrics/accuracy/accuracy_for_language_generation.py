# coding=utf-8
# Copyright 2021 Open Business Software Solutions, The HuggingFace Datasets Authors.
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
Accuracy metric. The part of this file is adapted from HuggingFace's
datasets package implementation of Accuracy metric. See
https://github.com/huggingface/datasets/blob/master/metrics/accuracy/accuracy.py
"""

from collections import Counter
from typing import Callable

import datasets
import numpy as np

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import normalize_text

_CITATION = """\
@inproceedings{papineni2002bleu,
  title={Bleu: a method for automatic evaluation of machine translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={Proceedings of the 40th annual meeting of the Association for Computational Linguistics},
  pages={311--318},
  year={2002}
}
"""

_DESCRIPTION = """
This metric is derived from Modified Unigram Precision as an accuracy metric so that
it will compute across maximum of reference or prediction. The computation is similar 
to precision and recall, however, we call this accuracy since there is no measure 
called "modified unigram accuracy".
Accuracy is the fraction of the common unigrams between the prediction
and the references among the max of prediction or reference tokens. It can be computed with:
Accuracy = # of matching tokens / max(# of prediction tokens, # of reference tokens)
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    'score': Accuracy score.
Examples:

    >>> accuracy = jury.load_metric("accuracy")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = accuracy.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': {'score': 0.7285714285714285}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AccuracyForLanguageGeneration(MetricForLanguageGeneration):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html"],
        )

    @staticmethod
    def _tokenize(predictions: Collator, references: Collator):
        predictions = [normalize_text(p).split() for p in predictions]
        references = [normalize_text(r).split() for r in references]
        return predictions, references

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        scores = []
        predictions, references = self._tokenize(predictions, references)
        for pred, ref in zip(predictions, references):
            score = 0
            pred_counts = Counter(pred)
            ref_counts = Counter(ref)
            for token, pred_count in pred_counts.items():
                if token in ref_counts:
                    score += min(pred_count, ref_counts[token])  # Intersection count
            scores.append(score / max(len(pred), len(ref)))
        avg_score = sum(scores) / len(scores)
        return {"score": avg_score}

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        scores = []
        for pred, refs in zip(predictions, references):
            pred_score = [
                self._compute_single_pred_single_ref(Collator([pred], keep=True), Collator([ref], keep=True))
                for ref in refs
            ]
            reduced_score = self._reduce_scores(pred_score, reduce_fn=reduce_fn)
            scores.append(reduced_score)

        return self._reduce_scores(scores, reduce_fn=np.mean)

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                pred_score = self._compute_single_pred_multi_ref(
                    Collator([pred], keep=True), Collator([refs], keep=True), reduce_fn=reduce_fn
                )
                pred_scores.append(pred_score)
            reduced_score = self._reduce_scores(pred_scores, reduce_fn=reduce_fn)
            scores.append(reduced_score)

        return self._reduce_scores(scores, reduce_fn=np.mean)
