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
SacreBLEU metric. The part of this file is adapted from SacreBLEU implementation
of datasets package. See
https://github.com/huggingface/datasets/blob/master/metrics/sacrebleu/sacrebleu.py
"""

import math
from typing import Callable, Dict, Sequence

import datasets
from packaging import version

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import PackagePlaceholder, get_token_lengths, requirement_message
from jury.tokenizer import BLEUDefaultTokenizer

# `import sacrebleu as scb` placeholder
scb = PackagePlaceholder(version="2.0.0")

__class_names__ = {"sacrebleu": "Sacrebleu"}


_CITATION = """\
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""

_DESCRIPTION = """\
SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

See the [README.md] file at https://github.com/mjpost/sacreBLEU for more information.
"""

_KWARGS_DESCRIPTION = """
Produces BLEU scores along with its sufficient statistics
from a source against one or more references.

Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    smooth_method: The smoothing method to use. (Default: 'exp').
    smooth_value: The smoothing value. Only valid for 'floor' and 'add-k'. (Defaults: floor: 0.1, add-k: 1).
    tokenize: Tokenization method to use for BLEU. If not provided, defaults to 'zh' for Chinese, 'ja-mecab' for
        Japanese and '13a' (mteval) otherwise.
    lowercase: Lowercase the data. If True, enables case-insensitivity. (Default: False).
    force: Insist that your tokenized input is actually detokenized.

Returns:
    'score': BLEU score,
    'counts': Counts,
    'totals': Totals,
    'precisions': Precisions,
    'bp': Brevity penalty,
    'sys_len': predictions length,
    'ref_len': reference length,
    'adjusted_precision': adjusted precisions with corrections for multiple predictions cases.

Examples:

    >>> sacrebleu = jury.load_metric("sacrebleu")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = sacrebleu.compute(predictions=predictions, references=references)
    >>> print(results)
    {'sacrebleu': {'score': 0.32377227131456443, 'counts': [11, 6, 3, 0], 'totals': [13, 11, 9, 7],
         'precisions': [0.8461538461538461, 0.5454545454545454, 0.33333333333333337, 0.07142857142857144], 
         'bp': 1.0, 'sys_len': 11, 'ref_len': 12, 
         'adjusted_precisions': [0.8461538461538461, 0.5454545454545454, 0.33333333333333337, 0.07142857142857144]}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SacrebleuForLanguageGeneration(MetricForLanguageGeneration):
    def _download_and_prepare(self, dl_manager):
        global scb
        try:
            import sacrebleu as scb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(metric_name="Sacrebleu", package_name="sacrebleu"))
        else:
            super(SacrebleuForLanguageGeneration, self)._download_and_prepare(dl_manager)

    def _info(self):
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `sacrebleu`, the module `sacrebleu>=1.4.12` is required, and the current version of `sacrebleu` doesn't match this condition.\n"
                'You can install it with `pip install "sacrebleu>=1.4.12"`.'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/mjpost/sacreBLEU",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/mjpost/sacreBLEU"],
            reference_urls=[
                "https://github.com/mjpost/sacreBLEU",
                "https://en.wikipedia.org/wiki/BLEU",
                "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",
            ],
        )

    def _tokenize(self, seq: Sequence[str]) -> Sequence[Sequence[str]]:
        tokenizer = BLEUDefaultTokenizer()
        return [tokenizer.tokenize(s) for s in seq]

    def _validate_references(self, references: Collator) -> None:
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError("Sacrebleu requires the same number of references for each prediction")

    def _compute_single_pred_single_ref(
        self,
        predictions: Collator,
        references: Collator,
        reduce_fn: Callable = None,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        tokenize=None,
        use_effective_order=False,
    ):
        output = scb.corpus_bleu(
            predictions,
            references,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            force=force,
            lowercase=lowercase,
            use_effective_order=use_effective_order,
            **(dict(tokenize=tokenize) if tokenize else {}),
        )
        output_dict = {
            "score": output.score / 100,
            "counts": output.counts,
            "totals": output.totals,
            "precisions": [p / 100 for p in output.precisions],
            "bp": output.bp,
            "sys_len": output.sys_len,
            "ref_len": output.ref_len,
        }
        return output_dict

    def _compute_single_pred_multi_ref(
        self,
        predictions: Collator,
        references: Collator,
        reduce_fn: Callable = None,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        tokenize=None,
        use_effective_order=False,
    ):
        # SacreBleu inherently supports multiple references.
        return self._compute_single_pred_single_ref(
            predictions=predictions,
            references=references,
            reduce_fn=reduce_fn,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            force=force,
            lowercase=lowercase,
            tokenize=tokenize,
            use_effective_order=use_effective_order,
        )

    def _compute_multi_pred_multi_ref(
        self,
        predictions: Collator,
        references: Collator,
        reduce_fn: Callable = None,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        tokenize=None,
        use_effective_order=False,
    ):
        flattened_predictions = []
        matched_references = []
        adjusted_prediction_length = 0
        for preds, refs in zip(predictions, references):
            n_preds = len(preds)
            tokenized_preds = self._tokenize(preds)
            adjusted_prediction_length += get_token_lengths(tokenized_preds, reduce_fn=max)
            flattened_predictions.extend([pred for pred in preds])
            matched_references.extend([refs] * n_preds)
        flattened_predictions = Collator(flattened_predictions, keep=True)
        matched_references = Collator(matched_references, keep=True)
        score = self._compute_single_pred_multi_ref(
            predictions=flattened_predictions,
            references=matched_references,
            reduce_fn=reduce_fn,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            force=force,
            lowercase=lowercase,
            tokenize=tokenize,
            use_effective_order=use_effective_order,
        )
        prediction_length, reference_length = score["sys_len"], score["ref_len"]
        ratio = prediction_length / reference_length
        adjusted_ratio = adjusted_prediction_length / reference_length
        if ratio > 1.0:
            adjusted_bp = 1.0
            scb_score = score["score"]
        else:
            bp = math.exp(1 - 1.0 / ratio)
            adjusted_bp = math.exp(1 - 1.0 / adjusted_ratio)
            scb_score = score["score"] * (adjusted_bp / bp)

        scb_score = scb_score * n_preds
        precisions = [p * n_preds for p in score["precisions"]]

        score.update(
            {
                "score": scb_score,
                "adjusted_precisions": precisions,
                "bp": adjusted_bp,
                "sys_len": adjusted_prediction_length,
            }
        )

        return score

    def evaluate(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ) -> Dict[str, float]:
        if predictions.can_collapse() and references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref
        self._validate_references(references)
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
