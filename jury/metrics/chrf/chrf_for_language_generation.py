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
ChrF(++) (Character n-gram F-score) metric. The part of this file is adapted from HuggingFace's
evaluate package implementation of CHRF metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/chrf/chrf.py
"""
from typing import Callable, Dict, List, Tuple, Union

import evaluate
from packaging import version

from jury.collator import Collator
from jury.metrics import LanguageGenerationInstance, MetricForLanguageGeneration
from jury.metrics._core.utils import PackagePlaceholder, requirement_message

# `import sacrebleu as scb` placeholder
scb = PackagePlaceholder(version="2.0.0")


_CITATION = """\
@inproceedings{popovic-2015-chrf,
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-3049",
    doi = "10.18653/v1/W15-3049",
    pages = "392--395",
}
@inproceedings{popovic-2017-chrf,
    title = "chr{F}++: words helping character n-grams",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Second Conference on Machine Translation",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-4770",
    doi = "10.18653/v1/W17-4770",
    pages = "612--618",
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
ChrF and ChrF++ are two MT evaluation metrics. They both use the F-score statistic for character n-gram matches,
and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment. We use the implementation
that is already present in sacrebleu.
The implementation here is slightly different from sacrebleu in terms of the required input format. The length of
the references and hypotheses lists need to be the same, so you may need to transpose your references compared to
sacrebleu's required input format. See https://github.com/huggingface/evaluate/issues/3154#issuecomment-950746534
See the README.md file at https://github.com/mjpost/sacreBLEU#chrf--chrf for more information.
"""

_KWARGS_DESCRIPTION = """
Produces ChrF(++) scores for hypotheses given reference translations.
Args:
    predictions: The system stream (a sequence of segments).
    references: A list of one or more reference streams (each a sequence of segments).
    char_order: Character n-gram order.
    word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
    beta: Determine the importance of recall w.r.t precision.
    lowercase: Enable case-insensitivity.
    whitespace: If `True`, include whitespaces when extracting character n-grams.
    eps_smoothing: If `True`, applies epsilon smoothing similar
    to reference chrF++.py, NLTK and Moses implementations. Otherwise,
    it takes into account effective match order similar to sacreBLEU < 2.0.0.
Returns:
    'score': The chrF (chrF++) score,
    'char_order': The character n-gram order,
    'word_order': The word n-gram order. If equals to 2, the metric is referred to as chrF++,
    'beta': Determine the importance of recall w.r.t precision
Examples:
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> chrf = jury.load_metric("chrf")
    >>> results = chrf.compute(predictions=prediction, references=reference)
    >>> print(results)
    {'chrf': {'score': 0.29778203723986857, 'char_order': 6, 'word_order': 0, 'beta': 2}}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CHRFForLanguageGeneration(MetricForLanguageGeneration):
    def _info(self):
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `sacrebleu`, the module `sacrebleu>=1.4.12` is required, and the current version of "
                "`sacrebleu` doesn't match this condition.\nYou can install it with `pip install sacrebleu>=1.4.12`."
            )
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/mjpost/sacreBLEU#chrf--chrf",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/mjpost/sacreBLEU#chrf--chrf"],
            reference_urls=[
                "https://github.com/m-popovic/chrF",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        global scb
        global CHRFScorer
        try:
            from sacrebleu import CHRF as CHRFScorer
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="chrf", package_name="sacrebleu"))
        else:
            super(CHRFForLanguageGeneration, self)._download_and_prepare(dl_manager)

    def _validate_references(self, references: Collator) -> None:
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError("Sacrebleu requires the same number of references for each prediction")

    def _compute_chrf_score(
        self, predictions: Union[str, List[str]], references: Union[str, List[str]], **kwargs
    ) -> Tuple[float, int, int, int]:
        if kwargs.get("char_order") is None:
            kwargs["char_order"] = CHRFScorer.CHAR_ORDER
        if kwargs.get("word_order") is None:
            kwargs["word_order"] = CHRFScorer.WORD_ORDER
        if kwargs.get("beta") is None:
            kwargs["beta"] = CHRFScorer.BETA
        sb_chrf = CHRFScorer(**kwargs)
        output = sb_chrf.corpus_score(predictions, references)

        return (
            output.score / 100,
            output.char_order,
            output.word_order,
            output.beta,
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        char_order: int = None,
        word_order: int = None,
        beta: int = None,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
    ):
        score, c_ord, w_ord, beta = self._compute_chrf_score(
            predictions,
            references,
            char_order=char_order,
            word_order=word_order,
            beta=beta,
            lowercase=lowercase,
            whitespace=whitespace,
            eps_smoothing=eps_smoothing,
        )
        return {"score": score, "char_order": c_ord, "word_order": w_ord, "beta": beta}

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        char_order: int = None,
        word_order: int = None,
        beta: int = None,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
    ):
        # SacreBleu inherently supports multiple references.
        return self._compute_single_pred_single_ref(
            predictions=predictions,
            references=references,
            reduce_fn=reduce_fn,
            char_order=char_order,
            word_order=word_order,
            beta=beta,
            lowercase=lowercase,
            whitespace=whitespace,
            eps_smoothing=eps_smoothing,
        )

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        char_order: int = None,
        word_order: int = None,
        beta: int = None,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
    ):
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                score, _, _, _ = self._compute_chrf_score(
                    pred,
                    refs,
                    char_order=char_order,
                    word_order=word_order,
                    beta=beta,
                    lowercase=lowercase,
                    whitespace=whitespace,
                    eps_smoothing=eps_smoothing,
                )
                pred_scores.append(score)
            scores.append(float(reduce_fn(pred_scores)))

        return {
            "score": sum(scores) / len(scores),
            "char_order": char_order or CHRFScorer.CHAR_ORDER,
            "word_order": word_order or CHRFScorer.WORD_ORDER,
            "beta": beta or CHRFScorer.BETA,
        }

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
