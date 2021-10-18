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
BLEU metric. The part of this file is adapted from HuggingFace's
datasets package implementation of BLEU metric. See
https://github.com/huggingface/datasets/blob/master/metrics/bleu/bleu.py
"""

import math
import os
from typing import Callable, Dict, Tuple

import datasets

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import download, get_token_lengths
from jury.tokenizer import BLEUDefaultTokenizer, TokenizerWrapper

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
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been 
machine-translated from one natural language to another. Quality is considered to be the correspondence between a 
machine's output and that of a human: "the closer a machine translation is to a professional human translation, the 
better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation 
with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.

Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good 
quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the 
translation's overall quality. Intelligibility or grammatical correctness are not taken into account[citation needed].

BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to 
the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a 
score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this 
reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional
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
    'score': BLEU score,
    'precisions': geometric mean of n-gram precisions,
    'brevity_penalty': brevity penalty,
    'length_ratio': ratio of lengths,
    'translation_length': translation_length,
    'reference_length': reference_length
Examples:

    >>> bleu = jury.load_metric("bleu")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = bleu.compute(predictions=predictions, references=references)
    >>> print(results)
    {'bleu': {'score': 0.42370250917168295, 
        'precisions': [0.8823529411764706, 0.6428571428571429, 0.45454545454545453, 0.125], 
        'brevity_penalty': 1.0, 'length_ratio': 1.0, 'translation_length': 11, 'reference_length': 11}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BleuForLanguageGeneration(MetricForLanguageGeneration):
    def __init__(self, resulting_name: str = None, compute_kwargs: Dict = None):
        self.should_change_resulting_name = True if resulting_name is None else False
        tokenizer = compute_kwargs.get("tokenizer", None) if compute_kwargs is not None else None
        self.tokenizer = BLEUDefaultTokenizer() if tokenizer is None else tokenizer
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs)

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
        of BLEU computation from tensorflow/nmt. See
        https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
        """
        nmt_source = "https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/scripts/bleu.py"
        nmt_dest = os.path.join(self.data_dir, "nmt_bleu.py")
        download(
            source=nmt_source,
            destination=nmt_dest,
        )
        self.external_module_path = nmt_dest

    def _tokenize(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        return tokenizer_wrapper.tokenize(predictions, references)

    def _compute_bleu_score(self, predictions: Collator, references: Collator, max_order=4, smooth=False):
        evaluation_fn = self._get_external_resource("nmt_bleu", attr="compute_bleu")
        score = evaluation_fn(
            reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth
        )
        (bleu, precisions, bp, ratio, translation_length, reference_length) = score
        return {
            "score": bleu,
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
        # Bleu score inherently supports multiple references.
        return self._compute_single_pred_single_ref(
            predictions=predictions, references=references, reduce_fn=reduce_fn, max_order=max_order, smooth=smooth
        )

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, max_order=4, smooth=False
    ):
        flattened_predictions = []
        matched_references = []
        adjusted_reference_length = adjusted_prediction_length = 0
        for preds, refs in zip(predictions, references):
            n_preds = len(preds)
            adjusted_reference_length += get_token_lengths(refs, reduce_fn=min)
            adjusted_prediction_length += get_token_lengths(preds, reduce_fn=max)
            flattened_predictions.extend([pred for pred in preds])
            matched_references.extend([refs] * n_preds)

        flattened_predictions = Collator(flattened_predictions, keep=True)
        matched_references = Collator(matched_references, keep=True)
        score = self._compute_single_pred_multi_ref(predictions=flattened_predictions, references=matched_references)

        prediction_length, reference_length = score["translation_length"], score["reference_length"]
        ratio = prediction_length / reference_length
        adjusted_ratio = adjusted_prediction_length / adjusted_reference_length

        bleu_score = score["score"]
        if ratio > 1.0:
            adjusted_bp = 1.0
            bleu_score = bleu_score
        else:
            bp = math.exp(1 - 1.0 / ratio)
            adjusted_bp = math.exp(1 - 1.0 / adjusted_ratio)
            bleu_score = bleu_score * (adjusted_bp / bp)

        bleu_score *= adjusted_ratio ** 2 / ratio

        score.update(
            {
                "score": bleu_score,
                "precisions": score["precisions"],
                "brevity_penalty": adjusted_bp,
                "length_ratio": adjusted_ratio,
                "translation_length": adjusted_prediction_length,
                "reference_length": adjusted_reference_length,
            }
        )

        return score

    def evaluate(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ) -> Dict[str, float]:
        max_order = kwargs.get("max_order")
        if max_order is not None and self.should_change_resulting_name:
            self.resulting_name += f"_{max_order}"
        if predictions.can_collapse() and references.can_collapse():
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref
        predictions, references = self._tokenize(predictions=predictions, references=references)
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
