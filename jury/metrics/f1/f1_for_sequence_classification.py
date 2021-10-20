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
"""
F1 metric. The part of this file is adapted from HuggingFace's
datasets package implementation of F1 metric. See
https://github.com/huggingface/datasets/blob/master/metrics/f1/f1.py
"""
from typing import List, Union

import datasets
from sklearn.metrics import f1_score

from jury.metrics._core import MetricForSequenceClassification, SequenceClassificationInstance, load_metric

_DESCRIPTION = """
The F1 score is the harmonic mean of the precision and recall. It can be computed with:
F1 = 2 * (precision * recall) / (precision + recall)
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
    f1: F1 score.
Examples:

    >>> f1_metric = datasets.load_metric("f1")
    >>> results = f1_metric.compute(predictions=[0, 1], references=[0, 1])
    >>> print(results)
    {'f1': 1.0}

    >>> predictions = [0, 2, 1, 0, 0, 1]
    >>> references = [0, 1, 2, 0, 1, 2]
    >>> results = f1_metric.compute(predictions=predictions, references=references, average="macro")
    >>> print(results)
    {'f1': 0.26666666666666666}
    >>> results = f1_metric.compute(predictions=predictions, references=references, average="micro")
    >>> print(results)
    {'f1': 0.3333333333333333}
    >>> results = f1_metric.compute(predictions=predictions, references=references, average="weighted")
    >>> print(results)
    {'f1': 0.26666666666666666}
    >>> results = f1_metric.compute(predictions=predictions, references=references, average=None)
    >>> print(results)
    {'f1': array([0.8, 0. , 0. ])}
"""

_CITATION = """
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
class F1ForSequenceClassification(MetricForSequenceClassification):
    _precision = load_metric("precision", task="sequence-classification")
    _recall = load_metric("recall", task="sequence-classification")

    def _info(self):
        return datasets.MetricInfo(
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
        labels=None,
        pos_label=1,
        average=None,
        sample_weight=None,
    ):
        score = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        return {"score": float(score) if score.size == 1 else score.tolist()}

    def _compute_single_pred_multi_ref(
        self, predictions: SequenceClassificationInstance, references: SequenceClassificationInstance, **kwargs
    ):
        predictions = predictions.nested()
        return self._compute_multi_pred_multi_ref(predictions=predictions, references=references, **kwargs)

    def _compute_multi_pred_multi_ref(
        self, predictions: SequenceClassificationInstance, references: SequenceClassificationInstance, **kwargs
    ):
        precision = self._precision.compute(predictions=predictions, references=references, **kwargs)
        recall = self._recall.compute(predictions=predictions, references=references, **kwargs)
        precision_scores = precision["precision"]["score"]
        recall_scores = recall["recall"]["score"]
        f1 = self._compute_f1(precision_scores, recall_scores)
        return {"score": f1}

    @staticmethod
    def _harmonic_mean(p, r):
        return 2 * p * r / (p + r)

    def _compute_f1(
        self, precision: Union[float, List[float]], recall: Union[float, List[float]]
    ) -> Union[float, List[float]]:
        if isinstance(precision, list) and isinstance(recall, list):
            return [self._harmonic_mean(p, r) for p, r in zip(precision, recall)]

        return self._harmonic_mean(precision, recall)
