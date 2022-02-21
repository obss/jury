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
""" CER metric. The part of this file is adapted from HuggingFace's
datasets package implementation of CER metric. See
https://github.com/huggingface/datasets/blob/master/metrics/cer/cer.py
"""
import warnings
from typing import Callable, List, Tuple, Union

import datasets
import jiwer
import jiwer.transforms as tr

from jury.metrics import LanguageGenerationInstance, MetricForLanguageGeneration

_CITATION = """\
@inproceedings{inproceedings,
    author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
    year = {2004},
    month = {01},
    pages = {},
    title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
"""

_DESCRIPTION = """\
Character error rate (CER) is a common metric of the performance of an automatic speech recognition system.
CER is similar to Word Error Rate (WER), but operates on character instead of word. Please refer to docs of WER for further information.
Character error rate can be computed as:
CER = (S + D + I) / N = (S + D + I) / (S + D + C)
where
S is the number of substitutions,
D is the number of deletions,
I is the number of insertions,
C is the number of correct characters,
N is the number of characters in the reference (N=S+D+C).
CER's output is not always a number between 0 and 1, in particular when there is a high number of insertions. This value is often associated to the percentage of characters that were incorrectly predicted. The lower the value, the better the
performance of the ASR system with a CER of 0 being a perfect score.
"""

_KWARGS_DESCRIPTION = """
Computes CER score of transcribed segments against references.
Args:
    references: list of references for each speech input.
    predictions: list of transcribtions to score.
    concatenate_texts: Whether or not to concatenate sentences before evaluation, set to True for more accurate result.
Returns:
    (float): the character error rate
Examples:
    >>> predictions = ["this is the prediction", "there is an other sample"]
    >>> references = ["this is the reference", "there is another one"]
    >>> cer = jury.load_metric("cer")
    >>> cer_score = cer.compute(predictions=predictions, references=references)
    >>> print(cer_score)
    0.34146341463414637
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CERForLanguageGeneration(MetricForLanguageGeneration):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
                "https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates",
            ],
        )

    @staticmethod
    def _get_cer_transform():
        return tr.Compose(
            [
                tr.RemoveMultipleSpaces(),
                tr.Strip(),
                tr.ReduceToSingleSentence(""),
                tr.ReduceToListOfListOfChars(),
            ]
        )

    def _compute_cer_score(
        self, predictions: Union[str, List[str]], references: Union[str, List[str]]
    ) -> Tuple[float, int, int, int, int]:
        cer_transform = self._get_cer_transform()
        measures = jiwer.compute_measures(
            references,
            predictions,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
        return (
            measures["wer"],
            int(measures["substitutions"]),
            int(measures["deletions"]),
            int(measures["insertions"]),
            int(measures["hits"]),
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        concatenate_texts: bool = False,
    ):
        if concatenate_texts:
            score, total_substitutions, total_deletions, total_insertions, total_hits = self._compute_cer_score(
                predictions, references
            )
        else:
            incorrect = 0
            total = 0
            total_substitutions = 0
            total_deletions = 0
            total_insertions = 0
            total_hits = 0
            for prediction, reference in zip(predictions, references):
                _, substitutions, deletions, insertions, hits = self._compute_cer_score(prediction, reference)
                total_substitutions += substitutions
                total_deletions += deletions
                total_insertions += insertions
                total_hits += hits
                incorrect += substitutions + deletions + insertions
                total += substitutions + deletions + hits
            score = incorrect / total

        return {
            "score": score,
            "overall": {
                "substitutions": total_substitutions,
                "deletions": total_deletions,
                "insertions": total_insertions,
                "hits": total_hits,
            },
        }

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs
    ):
        if "concatenate_texts" in kwargs:
            warnings.warn("Option 'concatenate_texts' is only available in single-pred & single-ref setting.")

        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_hits = 0
        total_refs = 0
        scores = []
        for pred, refs in zip(predictions, references):
            pred_scores = []
            for ref in refs:
                score, substitutions, deletions, insertions, hits = self._compute_cer_score(
                    predictions=pred, references=ref
                )
                pred_scores.append(score)
                total_substitutions += substitutions
                total_deletions += deletions
                total_insertions += insertions
                total_hits += hits
                total_refs += 1
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)

        return {
            "score": float(reduce_fn(scores)),
            "overall": {
                "substitutions": total_substitutions / total_refs,
                "deletions": total_deletions / total_refs,
                "insertions": total_insertions / total_refs,
                "hits": total_hits / total_refs,
            },
        }

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs
    ):
        if "concatenate_texts" in kwargs:
            warnings.warn("Option 'concatenate_texts' is only available in single-pred & single-ref setting.")

        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_hits = 0
        total_refs = 0
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                for ref in refs:
                    score, substitutions, deletions, insertions, hits = self._compute_cer_score(
                        predictions=pred, references=ref
                    )
                    pred_scores.append(score)
                    total_substitutions += substitutions
                    total_deletions += deletions
                    total_insertions += insertions
                    total_hits += hits
                    total_refs += 1
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)

        return {
            "score": float(reduce_fn(scores)),
            "overall": {
                "substitutions": total_substitutions / total_refs,
                "deletions": total_deletions / total_refs,
                "insertions": total_insertions / total_refs,
                "hits": total_hits / total_refs,
            },
        }


if __name__ == "__main__":
    predictions = ["this is the prediction", "there is an other sample", "a b c d"]
    references = [["this is the reference"], ["there is another one"], ["a d e f g y", "a b c d"]]

    cer = CERForLanguageGeneration()
    res = cer._compute_single_pred_multi_ref(predictions=predictions, references=references, concatenate_texts=False)
    print(res)
