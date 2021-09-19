# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
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
""" BLEU metric. """
import importlib.util

import datasets
from typing import Dict

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
                    "predictions": datasets.Sequence(datasets.Value("string"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string"), id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/BLEU",
                "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        nmt_source = "https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/scripts/bleu.py"
        nmt_dest = "metrics/bleu/nmt_bleu.py"
        module_path = download(nmt_source, nmt_dest)
        spec = importlib.util.spec_from_file_location("nmt_bleu", module_path)
        nmt_bleu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nmt_bleu)
        self.compute_bleu = nmt_bleu.compute_bleu

    def _preprocess(self, predictions, references):
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        predictions, references = tokenizer_wrapper.tokenize(predictions, references)
        if predictions.ndim > 2:
            predictions = predictions.reshape_len(-1)

        if references.ndim == 3:
            ref_count = references.shape[0]
            references = references.reshape(1, ref_count, -1)
        else:
            references = references.reshape(1, -1)

        return predictions, references

    def _compute_single_pred_single_ref(self, predictions: Collator, references: Collator, max_order=4, smooth=False):
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


if __name__ == "__main__":
    predictions = [[
        "It is a guide to action which ensures that the military always obeys the commands of the party"
    ]]
    references = [[
        "It is a guide to action that ensures that the military will forever heed Party commands"
    ]]

    # references = [[
    #     "It is a guide to action that ensures that the military will forever heed Party commands",
    #     "It is a guide to action which ensures that the military will forever heed Party commands"
    # ]]

    # Multi pred multi ref
    # predictions = [[
    #     "It is a guide to action which ensures that the military always obeys the commands of the party",
    #     "It is a guide to action that will ensure that the military always obeys the commands of the party"
    # ]]
    # references = [[
    #     "It is a guide to action that ensures that the military will forever heed Party commands",
    #     "It is a guide to action which ensures that the military will forever heed Party commands"
    # ]]
    bleu = Bleu()
    score = bleu.compute(predictions=predictions, references=references, use_aggregator=False)
    print(score)

