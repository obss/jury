# coding=utf-8
# Copyright 2021 Open Business Software Solutions, The HuggingFace evaluate Authors.
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
evaluate package implementation of Accuracy metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/accuracy/accuracy.py
"""
from typing import Callable

import evaluate
from sklearn.metrics import accuracy_score

from jury.metrics._core import MetricForSequenceClassification, SequenceClassificationInstance

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

_DESCRIPTION = """
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels, as returned by a model.
    references: Ground truth labels.
    normalize: If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight: Sample weights.
Returns:
    accuracy: Accuracy score.
Examples:

    >>> accuracy = jury.load_metric("accuracy", task="sequence-classification")
    >>> predictions = [[0], [2], [1], [0], [0], [1]]
    >>> references = [[0], [1], [2], [0], [1], [2]]
    >>> results = accuracy.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': {'score': 0.3333333333333333}}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AccuracyForSequenceClassification(MetricForSequenceClassification):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html"],
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        normalize=True,
        sample_weight=None,
    ):
        accuracy = float(
            accuracy_score(y_pred=predictions, y_true=references, normalize=normalize, sample_weight=sample_weight)
        )
        return {"score": accuracy}

    def _compute_single_pred_multi_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        exact_match=True,
    ):
        n_samples = len(predictions)
        match_sum = 0
        for pred, ref in zip(predictions, references):
            if exact_match and [pred] == ref:
                match_sum += 1
            elif not exact_match and pred in ref:
                match_sum += 1
        return {"score": match_sum / n_samples}

    def _compute_multi_pred_multi_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        exact_match=True,
    ):
        n_samples = len(predictions)
        match_sum = 0
        for preds, ref in zip(predictions, references):
            if exact_match and preds == ref:
                match_sum += 1
            else:
                for pred in preds:
                    if pred in ref:
                        match_sum += 1
                        break
        return {"score": match_sum / n_samples}
