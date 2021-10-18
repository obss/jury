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
F1 metric. The part of this file is adapted from HuggingFace's
datasets package implementation of F1 metric. See
https://github.com/huggingface/datasets/blob/master/metrics/f1/f1.py
"""

from typing import Callable

import datasets
import numpy as np

from jury.collator import Collator
from jury.metrics._core import LanguageGenerationInstance, MetricForLanguageGeneration, load_metric
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
Harmonic mean of precision and recall metrics. The precision and recall it uses 
are the implementations of `jury.metrics.precision` and `jury.metrics.recall` respectively.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    'score': F1 score.
Examples:

    >>> f1 = jury.load_metric("f1")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = f1.compute(predictions=predictions, references=references)
    >>> print(results)
    {'f1': {'score': 0.7948717948717947}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class F1ForLanguageGeneration(MetricForLanguageGeneration):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html"],
        )

    def _tokenize(self, predictions: LanguageGenerationInstance, references: LanguageGenerationInstance):
        predictions = [normalize_text(p).split() for p in predictions]
        references = [normalize_text(r).split() for r in references]
        return predictions, references

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        recall = load_metric("recall", task="language-generation")
        precision = load_metric("precision", task="language-generation")
        predictions, references = predictions.nested(), references.nested()
        recall_score = recall.compute(predictions=predictions, references=references)["recall"]["score"]
        precision_score = precision.compute(predictions=predictions, references=references)["precision"]["score"]
        try:
            f1 = (2 * recall_score * precision_score) / (recall_score + precision_score)
        except ZeroDivisionError:
            return {"score": 0.0}
        return {"score": f1}

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs
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
