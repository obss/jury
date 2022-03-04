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
""" Word Error Ratio (WER) metric. The part of this file is adapted from HuggingFace's
datasets package implementation of CER metric. See
https://github.com/huggingface/datasets/blob/master/metrics/wer/wer.py
"""
import warnings
from typing import Callable, List, Tuple, Union

import datasets

from jury.metrics import LanguageGenerationInstance, MetricForLanguageGeneration
from jury.metrics._core.utils import PackagePlaceholder, requirement_message

# `import sacrebleu as scb` placeholder
scb = PackagePlaceholder(version="2.0.0")


_CITATION = """\
@inproceedings{snover-etal-2006-study,
    title = "A Study of Translation Edit Rate with Targeted Human Annotation",
    author = "Snover, Matthew  and
      Dorr, Bonnie  and
      Schwartz, Rich  and
      Micciulla, Linnea  and
      Makhoul, John",
    booktitle = "Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers",
    month = aug # " 8-12",
    year = "2006",
    address = "Cambridge, Massachusetts, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2006.amta-papers.25",
    pages = "223--231",
}
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
TER (Translation Edit Rate, also called Translation Error Rate) is a metric to quantify the edit operations that a
hypothesis requires to match a reference translation. We use the implementation that is already present in sacrebleu
(https://github.com/mjpost/sacreBLEU#ter), which in turn is inspired by the TERCOM implementation, which can be found
here: https://github.com/jhclark/tercom.
The implementation here is slightly different from sacrebleu in terms of the required input format. The length of
the references and hypotheses lists need to be the same, so you may need to transpose your references compared to
sacrebleu's required input format. See https://github.com/huggingface/datasets/issues/3154#issuecomment-950746534
See the README.md file at https://github.com/mjpost/sacreBLEU#ter for more information.
"""

_KWARGS_DESCRIPTION = """
Produces TER scores alongside the number of edits and reference length.
Args:
    predictions: The system stream (a sequence of segments).
    references: A list of one or more reference streams (each a sequence of segments).
    normalized: Whether to apply basic tokenization to sentences.
    no_punct: Whether to remove punctuations from sentences.
    asian_support: Whether to support Asian character processing.
    case_sensitive: Whether to disable lowercasing.
Returns:
    'score': TER score (num_edits / sum_ref_lengths * 100),
    'num_edits': The cumulative number of edits,
    'ref_length': The cumulative average reference length.
Examples:
    >>> predictions = ["hello there general kenobi", "foo bar foobar"]
    >>> references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
    >>> ter = jury.load_metric("ter")
    >>> results = ter.compute(predictions=predictions, references=references)
    >>> print(results)
    {'score': 0.0, 'num_edits': 0, 'ref_length': 6.5}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TERForLanguageGeneration(MetricForLanguageGeneration):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="http://www.cs.umd.edu/~snover/tercom/",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/mjpost/sacreBLEU#ter"],
            reference_urls=[
                "https://github.com/jhclark/tercom",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        global scb
        global TERScorer

        try:
            import sacrebleu as scb
            from sacrebleu import TER as TERScorer
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="WER", package_name="jiwer"))
        else:
            super(TERForLanguageGeneration, self)._download_and_prepare(dl_manager)

    def _compute_ter_score(self, predictions: LanguageGenerationInstance, references: LanguageGenerationInstance, **kwargs):
        sb_ter = TERScorer(**kwargs)
        output = sb_ter.corpus_score(predictions, references)
        return {"score": output.score, "num_edits": output.num_edits, "ref_length": output.ref_length}

    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
    ):
        transformed_references = [[r] for r in references]
        return self._compute_ter_score(
                predictions=predictions,
                references=transformed_references,
                normalized=normalized,
                no_punct=no_punct,
                asian_support=asian_support,
                case_sensitive=case_sensitive
        )

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
    ):
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError("Sacrebleu requires the same number of references for each prediction")
        transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]
        # SacreBleu inherently supports multiple references.
        return self._compute_ter_score(
                predictions=predictions,
                references=transformed_references,
                normalized=normalized,
                no_punct=no_punct,
                asian_support=asian_support,
                case_sensitive=case_sensitive
        )

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
    ):
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                score = self._compute_single_pred_multi_ref(
                        predictions=[pred],
                        references=[refs],
                        normalized=normalized,
                        no_punct=no_punct,
                        asian_support=asian_support,
                        case_sensitive=case_sensitive
                )
                pred_scores.append(score["score"])
            pred_score = reduce_fn(pred_scores)
            scores.append(pred_score)
        return {"score": sum(scores) / len(scores)}


if __name__ == "__main__":
    import json

    predictions = [["hello there general kenobi", "hi there"], ["foo bar foobar"]]
    # references = ["hello there general kenobi", "foo bar foobar"]
    references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]

    ter = TERForLanguageGeneration()
    res = ter.compute(predictions=predictions, references=references)

    print(json.dumps(res, indent=2))
