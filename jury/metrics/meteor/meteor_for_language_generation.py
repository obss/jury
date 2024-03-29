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
METEOR metric. The part of this file is adapted from HuggingFace's
evaluate package implementation of METEOR metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/meteor/meteor.py
"""

from typing import Callable, Dict, Tuple

import evaluate
import numpy as np
from nltk import __version__ as NLTK_VERSION
from nltk.translate import meteor_score
from packaging.version import Version

if Version(NLTK_VERSION) < Version("3.6.6"):
    raise EnvironmentError(
        f"Version constraints does not hold for 'nltk', expected version >=3.6.6, got {NLTK_VERSION}."
    )

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration
from jury.tokenizer import DefaultTokenizer, TokenizerWrapper

_CITATION = """\
@inproceedings{banarjee2005,
  title     = {{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments},
  author    = {Banerjee, Satanjeev  and Lavie, Alon},
  booktitle = {Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization},
  month     = jun,
  year      = {2005},
  address   = {Ann Arbor, Michigan},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.aclweb.org/anthology/W05-0909},
  pages     = {65--72},
}
"""

_DESCRIPTION = """\
METEOR, an automatic metric for machine translation evaluation
that is based on a generalized concept of unigram matching between the
machine-produced translation and human-produced reference translations.
Unigrams can be matched based on their surface forms, stemmed forms,
and meanings; furthermore, METEOR can be easily extended to include more
advanced matching strategies. Once all generalized unigram matches
between the two strings have been found, METEOR computes a score for
this matching using a combination of unigram-precision, unigram-recall, and
a measure of fragmentation that is designed to directly capture how
well-ordered the matched words in the machine translation are in relation
to the reference.

METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic
data and 0.331 on the Chinese data. This is shown to be an improvement on
using simply unigram-precision, unigram-recall and their harmonic F1
combination.
"""

_KWARGS_DESCRIPTION = """
Computes METEOR score of translated segments against one or more references.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    alpha: Parameter for controlling relative weights of precision and recall. default: 0.9
    beta: Parameter for controlling shape of penalty as a function of fragmentation. default: 3
    gamma: Relative weight assigned to fragmentation penalty. default: 0.5
Returns:
    'score': meteor score.
Examples:

    >>> meteor = jury.load_metric("meteor")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = meteor.compute(predictions=predictions, references=references)
    >>> print(results)
    {'meteor': {'score': 0.5420511682934044}}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MeteorForLanguageGeneration(MetricForLanguageGeneration):
    def __init__(self, resulting_name: str = None, compute_kwargs: Dict = None, **kwargs):
        self.should_change_resulting_name = resulting_name is None
        self.tokenizer = DefaultTokenizer()
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/nltk/nltk/blob/develop/nltk/translate/meteor_score.py"],
            reference_urls=[
                "https://www.nltk.org/api/nltk.translate.html#module-nltk.translate.meteor_score",
                "https://en.wikipedia.org/wiki/METEOR",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    def _preprocess(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        return tokenizer_wrapper.tokenize(predictions, references)

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, alpha=0.9, beta=3, gamma=0.5
    ):
        scores = [
            meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
            for ref, pred in zip(references, predictions)
        ]
        return {"score": self._reduce_scores(scores, reduce_fn=np.mean)}

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, alpha=0.9, beta=3, gamma=0.5
    ):
        scores = [
            meteor_score.meteor_score(references=ref, hypothesis=pred, alpha=alpha, beta=beta, gamma=gamma)
            for ref, pred in zip(references, predictions)
        ]
        return {"score": self._reduce_scores(scores, reduce_fn=np.mean)}

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, alpha=0.9, beta=3, gamma=0.5
    ):
        scores = []
        for pred, ref in zip(predictions, references):
            score = [
                meteor_score.meteor_score(references=ref, hypothesis=p, alpha=alpha, beta=beta, gamma=gamma)
                for p in pred
            ]
            reduced_score = reduce_fn(score)
            scores.append(reduce_fn(reduced_score))
        return {"score": self._reduce_scores(scores, reduce_fn=np.mean)}
