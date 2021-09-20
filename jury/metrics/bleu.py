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
BLEU metric. The part of this file is adapted from HuggingFace's
datasets package implementation of BLEU metric. See
https://github.com/huggingface/datasets/blob/master/metrics/bleu/bleu.py
"""

import importlib.util
import math
import os
from typing import Callable, Dict, Iterable, List, Tuple, Union

import datasets
import numpy as np

from jury.collator import Collator
from jury.metrics._base import Metric
from jury.metrics._utils import download
from jury.tokenizer import BLEUDefaultTokenizer, TokenizerWrapper

__class_names__ = {"bleu": "BLEU"}


_CITATION = """\
@INPROCEEDINGS{Papineni02bleu:a,
    author = {Kishore Papineni and Salim Roukos and Todd Ward and Wei-jing Zhu},
    title = {BLEU: a Method for Automatic Evaluation of Machine Translation},
    booktitle = {},
    year = {2002},
    pages = {311--318}
}
@inproceedings{lin-och-2004-orange,
    title = "{ORANGE}: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation",
    author = "Lin, Chin-Yew  and
      Och, Franz Josef",
    booktitle = "{COLING} 2004: Proceedings of the 20th International Conference on Computational Linguistics",
    month = "aug 23{--}aug 27",
    year = "2004",
    address = "Geneva, Switzerland",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/C04-1072",
    pages = "501--507",
}
"""

_DESCRIPTION = """\
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation,
the better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and
remains one of the most popular automated and inexpensive metrics.

Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations.
Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness
are not taken into account[citation needed].

BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1
representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the
reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional
reference translations will increase the BLEU score.
"""

_KWARGS_DESCRIPTION = """
Computes BLEU score of translated segments against one or more references.
Args:
    predictions: list of translations to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
Returns:
    'bleu': bleu score,
    'precisions': geometric mean of n-gram precisions,
    'brevity_penalty': brevity penalty,
    'length_ratio': ratio of lengths,
    'translation_length': translation_length,
    'reference_length': reference_length
Examples:

    >>> predictions = [
    ...     ["hello", "there", "general", "kenobi"],                             # tokenized prediction of the first sample
    ...     ["foo", "bar", "foobar"]                                             # tokenized prediction of the second sample
    ... ]
    >>> references = [
    ...     [["hello", "there", "general", "kenobi"], ["hello", "there", "!"]],  # tokenized references for the first sample (2 references)
    ...     [["foo", "bar", "foobar"]]                                           # tokenized references for the second sample (1 reference)
    ... ]
    >>> bleu = datasets.load_metric("bleu")
    >>> results = bleu.compute(predictions=predictions, references=references)
    >>> print(results["bleu"])
    1.0
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Bleu(Metric):
    def __init__(self, resulting_name: str = None, params: Dict = None):
        resulting_name = "BLEU" if resulting_name is None else resulting_name
        tokenizer = params.get("tokenizer", None) if params is not None else None
        self.tokenizer = BLEUDefaultTokenizer() if tokenizer is None else tokenizer
        super().__init__(resulting_name=resulting_name, params=params)

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="tokens"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="tokens"), id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/BLEU",
                "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",
            ],
        )

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of bleu score from the implementation
        of BLEU computation from tensorflow/nmt module. See
        https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
        """
        nmt_source = "https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/scripts/bleu.py"
        nmt_dest = os.path.join(self.data_dir, "nmt_bleu.py")
        if not os.path.exists(nmt_dest):
            download(nmt_source, nmt_dest)
        spec = importlib.util.spec_from_file_location("nmt_bleu", nmt_dest)
        nmt_bleu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nmt_bleu)
        self.compute_bleu = nmt_bleu.compute_bleu

    def _tokenize(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        return tokenizer_wrapper.tokenize(predictions, references)

    @staticmethod
    def _get_token_lengths(sequences: Iterable, reduce_fn: Callable = None) -> Union[int, List[int]]:
        token_lengths = [len(item) for item in sequences]
        if reduce_fn is not None:
            return int(reduce_fn(token_lengths))
        return token_lengths

    def _compute_bleu_score(self, predictions: Collator, references: Collator, max_order=4, smooth=False):
        score = self.compute_bleu(
            reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth
        )
        (bleu, precisions, bp, ratio, translation_length, reference_length) = score
        return {
            "bleu": bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "translation_length": translation_length,
            "reference_length": reference_length,
        }

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, max_order=4, smooth=False
    ):
        predictions = predictions.reshape(
            len(predictions),
        )
        return self._compute_bleu_score(predictions=predictions, references=references)

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, max_order=4, smooth=False
    ):
        # Bleu score implementation can natively handle multiple references.
        return self._compute_single_pred_single_ref(
            predictions=predictions, references=references, reduce_fn=reduce_fn, max_order=max_order, smooth=smooth
        )

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, max_order=4, smooth=False
    ):
        flattened_predictions = []
        matched_references = []
        reference_length = prediction_length = adjusted_reference_length = adjusted_prediction_length = 0
        for preds, refs in zip(predictions, references):
            n_preds = len(preds)
            reference_length += self._get_token_lengths(refs, reduce_fn=min) * n_preds
            adjusted_reference_length += self._get_token_lengths(refs, reduce_fn=min)
            prediction_length += self._get_token_lengths(preds, reduce_fn=sum)
            adjusted_prediction_length += self._get_token_lengths(preds, reduce_fn=max)
            flattened_predictions.extend([pred for pred in preds])
            matched_references.extend([refs] * n_preds)

        flattened_predictions = Collator(flattened_predictions, keep=True)
        matched_references = Collator(matched_references, keep=True)
        score = self._compute_single_pred_multi_ref(predictions=flattened_predictions, references=matched_references)

        ratio = prediction_length / reference_length
        adjusted_ratio = adjusted_prediction_length / adjusted_reference_length
        if ratio > 1.0:
            adjusted_bp = 1.0
            bleu_score = score["bleu"]
        else:
            bp = math.exp(1 - 1.0 / ratio)
            adjusted_bp = math.exp(1 - 1.0 / adjusted_ratio)
            bleu_score = score["bleu"] * (adjusted_bp / bp)

        return {
            "bleu": bleu_score,
            "precisions": score["precisions"],
            "brevity_penalty": adjusted_bp,
            "length_ratio": adjusted_ratio,
            "translation_length": adjusted_prediction_length,
            "reference_length": adjusted_reference_length,
        }

    def evaluate(self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs) -> Dict[str, float]:
        if predictions.can_collapse() and references.can_collapse():
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref
        predictions, references = self._tokenize(predictions=predictions, references=references)
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
