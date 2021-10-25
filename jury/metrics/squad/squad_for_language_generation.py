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
""" SQuAD metric. The part of this file is adapted from SacreBLEU implementation
of datasets package. See
https://github.com/huggingface/datasets/blob/master/metrics/squad/squad.py"""
import os
from typing import Callable, Dict, List

import datasets
import numpy as np
import pandas as pd

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import download
from jury.utils import NestedSingleType

_CITATION = """\
@inproceedings{Rajpurkar2016SQuAD10,
  title={SQuAD: 100, 000+ Questions for Machine Comprehension of Text},
  author={Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy Liang},
  booktitle={EMNLP},
  year={2016}
}
"""

_DESCRIPTION = """
This metric wrap the official scoring script for version 1 of the Stanford Question Answering Dataset (SQuAD).

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by
crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span,
from the corresponding reading passage, or the question might be unanswerable.
"""

_KWARGS_DESCRIPTION = """
Computes SQuAD scores (F1 and EM).
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces. Optionally,
        List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair as given in the references (see below)
        - 'prediction_text': the text of the answer
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces. Optionally,
        List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair (see above),
        - 'answers': a Dict in the SQuAD dataset format
            {
                'text': list of possible texts for the answer, as a list of strings
                'answer_start': list of start positions for the answer, as a list of ints
            }
            Note that answer_start values are not taken into account to compute the metric.
Returns:
    'exact_match': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
Examples:

    >>> squad = jury.load_metric("squad")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = squad.compute(predictions=predictions, references=references)
    >>> print(results)
    {'squad': {'exact_match': 0.0, 'f1': 74.02597402597402}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SquadForLanguageGeneration(MetricForLanguageGeneration):
    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of squad score from the implementation
        of Squad computation from huggingface/datasets. See
        https://github.com/huggingface/datasets/blob/master/metrics/squad/evaluate.py
        """
        squad_source = "https://raw.githubusercontent.com/huggingface/datasets/master/metrics/squad/evaluate.py"
        squad_dest = os.path.join(self.data_dir, "squad_evaluate.py")
        download(
            source=squad_source,
            destination=squad_dest,
        )
        self.external_module_path = squad_dest

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
            reference_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
        )

    def _preprocess(self, predictions: Collator, references: Collator):
        if NestedSingleType.get_type(predictions, order=-1) == "str" and predictions.can_collapse():
            predictions = [{"prediction_text": pred, "id": str(i)} for i, pred in enumerate(predictions.collapse())]
        elif NestedSingleType.get_type(predictions, order=-1) == "str" and not predictions.can_collapse():
            _predictions = []
            for i, preds in enumerate(predictions):
                _predictions.append([{"prediction_text": pred, "id": str(i)} for pred in preds])
            predictions = _predictions
        if NestedSingleType.get_type(references, order=-1) == "str":
            references = [
                {"answers": {"answer_start": [-1], "text": Collator(ref).collapse()}, "id": str(i)}
                for i, ref in enumerate(references)
            ]
        return predictions, references

    def _compute_single_pred_single_ref(self, predictions: Collator, references: Collator, **kwargs):
        pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                                "id": ref["id"],
                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        evaluation_fn = self._get_external_resource("squad_evaluate", attr="evaluate")
        score = evaluation_fn(dataset=dataset, predictions=pred_dict)
        for metric_, score_ in score.items():
            score[metric_] = score_ / 100
        return score

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        return self._compute_single_pred_single_ref(predictions=predictions, references=references, reduce_fn=reduce_fn)

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                pred_scores.append(
                    self._compute_single_pred_multi_ref(
                        predictions=Collator([pred], keep=True),
                        references=Collator([refs], keep=True),
                        reduce_fn=reduce_fn,
                    )
                )
            reduced_score = self._reduce_multi_pred_scores(pred_scores, reduce_fn)
            scores.append(reduced_score)

        # Average reduced scores
        return self._reduce_multi_pred_scores(scores, np.mean)

    def _reduce_multi_pred_scores(self, results: List[Dict], reduce_fn) -> Dict:
        df = pd.DataFrame(results)
        return df.apply(reduce_fn, axis=0).to_dict()
