# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Precision metric."""
from collections import Counter
from typing import Callable, Dict

import datasets
import numpy as np

from jury.collator import Collator
from jury.metrics import Metric
from jury.metrics._utils import normalize_text

__class_names__ = {"recall": "Recall"}


_DESCRIPTION = """
Precision is the fraction of the true examples among the predicted examples. It can be computed with:
Precision = TP / (TP + FP)
TP: True positive
FP: False positive
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels, as returned by a model.
    references: Ground truth labels.
    labels: The set of labels to include when average != 'binary', and
        their order if average is None. Labels present in the data can
        be excluded, for example to calculate a multiclass average ignoring
        a majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in y_true and
        y_pred are used in sorted order.
    average: This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
            binary: Only report results for the class specified by pos_label.
                This is applicable only if targets (y_{true,pred}) are binary.
            micro: Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            macro: Calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            weighted: Calculate metrics for each label, and find their average
                weighted by support (the number of true instances for each label).
                This alters ‘macro’ to account for label imbalance; it can result
                in an F-score that is not between precision and recall.
            samples: Calculate metrics for each instance, and find their average
                (only meaningful for multilabel classification).
    sample_weight: Sample weights.
Returns:
    precision: Precision score.
Examples:

    >>> precision_metric = datasets.load_metric("precision")
    >>> results = precision_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'precision': 1.0}
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Recall(Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Sequence(datasets.Value("string")),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html"],
        )

    def _tokenize(self, predictions: Collator, references: Collator):
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
            scores.append(score / len(ref))
        avg_score = sum(scores) / len(scores)
        return {"score": avg_score}

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs
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

    def _compute_multi_pred_multi_ref(self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs):
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
