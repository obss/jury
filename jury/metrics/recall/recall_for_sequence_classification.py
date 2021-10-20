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
Recall metric. The part of this file is adapted from HuggingFace's
datasets package implementation of Recall metric. See
https://github.com/huggingface/datasets/blob/master/metrics/recall/recall.py
"""
from typing import Callable

import datasets
from sklearn.metrics import recall_score

from jury.metrics._core import MetricForSequenceClassification, SequenceClassificationInstance

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
    >>> predictions = [0, 2, 1, 0, 0, 1]
    >>> references = [0, 1, 2, 0, 1, 2]
    >>> results = precision_metric.compute(predictions=predictions, references=references, average='macro')
    >>> print(results)
    {'precision': 0.2222222222222222}
    >>> results = precision_metric.compute(predictions=predictions, references=references, average='micro')
    >>> print(results)
    {'precision': 0.3333333333333333}
    >>> results = precision_metric.compute(predictions=predictions, references=references, average='weighted')
    >>> print(results)
    {'precision': 0.2222222222222222}
    >>> results = precision_metric.compute(predictions=predictions, references=references, average=None)
    >>> print(results)
    {'precision': array([0.66666667, 0.        , 0.        ])}
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
class RecallForSequenceClassification(MetricForSequenceClassification):
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
        score = recall_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        return {"recall": float(score) if score.size == 1 else score.tolist()}

    def _compute_single_pred_multi_ref(
        self, predictions: SequenceClassificationInstance, references: SequenceClassificationInstance, **kwargs
    ):
        recalls = []
        labels = self._get_class_ids(references)
        for label in labels:
            n_samples = len([1 for sample in references if label in sample])
            match_sum = 0
            for pred, refs in zip(predictions, references):
                if label not in refs:
                    continue
                if pred in refs:
                    match_sum += 1
            label_precision = match_sum / n_samples
            recalls.append(label_precision)
        return {"score": recalls}

    def _compute_multi_pred_multi_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        exact_match=True,
    ):
        recalls = []
        labels = self._get_class_ids(references)
        for label in labels:
            n_samples = len([1 for sample in references if label in sample])
            match_sum = 0
            for preds, refs in zip(predictions, references):
                if label not in refs:
                    continue
                if set(refs).intersection(set(preds)):
                    match_sum += 1
            label_precision = match_sum / n_samples
            recalls.append(label_precision)
        return {"score": recalls}
