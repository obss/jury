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
Prism metric. The part of this file is adapted from metric implementations
of evaluate package. See
https://github.com/huggingface/evaluate/blob/master/metrics/
"""
import os
from typing import Callable, Dict, List, Union

import evaluate
import validators

from jury.metrics import LanguageGenerationInstance
from jury.metrics._core import MetricForLanguageGeneration

_CITATION = """
@inproceedings{thompson-post-2020-automatic,
    title={Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing},
    author={Brian Thompson and Matt Post},
    year={2020},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
"""

_DESCRIPTION = """
Prism is an automatic MT metric which uses a sequence-to-sequence paraphraser to score MT system outputs 
conditioned on their respective human references. Prism uses a multilingual NMT model as a zero-shot paraphraser, 
which negates the need for synthetic paraphrase data and results in a single model which works in many languages.

See the `README.md` file at [https://github.com/thompsonb/prism](https://github.com/thompsonb/prism) for more
information.
"""

_KWARGS_DESCRIPTION = """
Prism metric arguments.

Construction Args:
    model_path_or_url (str): Path to the model directory or a URL of model file (tar).
    lang (str): Language of the sentences; required (e.g. 'en').
    temperature (float): Softmax temperature, where values >1.0 produce more uniform samples 
        and values <1.0 produce sharper samples.
    
Computation Args:
    predictions (list of str): Prediction/candidate sentences.
    references (list of str or list of list of str): Reference sentences.
    segment_scores (bool): If True, then score for each instance are returned separately. Otherwise,
        average score is returned.
    normalized (bool): If True, resulting score/scores are normalized with exponentiation by log base,
        which bounds the score range within [0,1] (still higher is better). 

Returns:
    'score': Prism score.

Examples:

    >>> prism = jury.load_metric("prism")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = prism.compute(predictions=predictions, references=references)
    >>> print(results)
    {
      "prism": {
        "score": -2.3071975708007812,
        "identifier": {
          "version": "0.1",
          "model": "m39v1",
          "seg_scores": "avg_log_prob",
          "sys_scores": "avg_log_prob",
          "log_base": 2,
          "temperature": 1.0
        },
        "model_path_or_url": "default",
        "lang": "en",
        "segment_scores": false,
        "normalized": false
      }
    }
"""

_LICENSE = """MIT License

Copyright (c) Brian Thompson

Portions of this software are copied from fairseq (https://github.com/pytorch/fairseq),
which is released under the MIT License and Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CHECKPOINT_URLS = {
    "default": {
        "url": "https://huggingface.co/Devrim/prism-default/resolve/main/m39v1.tar",
        "model_dir": "m39v1",
    }
}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PrismForLanguageGeneration(MetricForLanguageGeneration):
    def __init__(
        self,
        resulting_name: str = None,
        compute_kwargs: Dict = None,
        model_path_or_url: str = "default",
        lang: str = "en",
        temperature: float = 1.0,
        **kwargs,
    ):
        self.model_path_or_url = model_path_or_url
        self.lang = lang
        self.temperature = temperature
        self.model_dir = None
        self.scorer = None
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @property
    def model_identifier(self):
        if self.scorer is not None:
            return self.scorer.identifier()

    def _download_model(self, dl_manager):
        if (
            self.model_path_or_url not in CHECKPOINT_URLS
            and not os.path.isdir(self.model_path_or_url)
            and not validators.url(self.model_path_or_url)
        ):
            raise ValueError("Provided 'model_path_or_url' neither points to an existing directory nor a valid URL.")
        elif os.path.isdir(self.model_path_or_url):
            self.model_dir = self.model_path_or_url
        else:
            if self.model_path_or_url in CHECKPOINT_URLS:
                model_source = CHECKPOINT_URLS[self.model_path_or_url]["url"]
                model_dir = CHECKPOINT_URLS[self.model_path_or_url]["model_dir"]
            else:
                model_source = self.model_path_or_url
                model_dir = os.path.basename(self.model_path_or_url).replace(".tar", "")
            extraction_dir = dl_manager.download_and_extract(model_source)
            self.model_dir = os.path.join(extraction_dir, model_dir)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of Prism score from the implementation
        of Prism computation from thompsonb/prism. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/thompsonb/prism/blob/42e45a46d1c7924e98bceeed2ea81b31efcb6f9d/prism.py
        """
        self._download_model(dl_manager)
        prism_source = (
            "https://raw.githubusercontent.com/thompsonb/prism/42e45a46d1c7924e98bceeed2ea81b31efcb6f9d/prism.py"
        )
        self.external_module_path = dl_manager.download(
            prism_source,
        )

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/thompsonb/prism",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/thompsonb/prism"],
            reference_urls=[
                "https://github.com/thompsonb/prism",
                "https://www.aclweb.org/anthology/2020.emnlp-main.8/",
            ],
            license=_LICENSE,
        )

    def _load_scorer(self):
        if self.scorer is None:
            Prism = self._get_external_resource("prism", attr="Prism")
            self.scorer = Prism(model_dir=self.model_dir, lang=self.lang, temperature=self.temperature)

    def _compute_prism_score(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        segment_scores: bool,
        **kwargs,
    ) -> Union[float, List[float]]:
        self._load_scorer()
        score = self.scorer.score(ref=references, cand=predictions, segment_scores=segment_scores, **kwargs)
        if segment_scores:
            return score.tolist()
        return float(score)

    @staticmethod
    def _normalize_score(score: Union[float, List[float]], exponent: float):
        if isinstance(score, float):
            return exponent ** score
        return [exponent ** s for s in score]

    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn=None,
        segment_scores: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        score = self._compute_prism_score(predictions, references, segment_scores=segment_scores, **kwargs)
        if normalize:
            score = self._normalize_score(score, self.model_identifier["log_base"])

        return {
            "score": score,
            "identifier": self.model_identifier,
            "model_path_or_url": self.model_path_or_url,
            "lang": self.lang,
            "segment_scores": segment_scores,
            "normalized": normalize,
        }

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        segment_scores: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        self._load_scorer()
        scores = []
        for pred, refs in zip(predictions, references):
            pred = [pred] * len(refs)
            prism_score = self._compute_prism_score(predictions=pred, references=refs, segment_scores=True, **kwargs)
            reduced_score = float(reduce_fn(prism_score))
            scores.append(reduced_score)

        if not segment_scores:
            scores = sum(scores) / len(scores)

        if normalize:
            scores = self._normalize_score(scores, self.model_identifier["log_base"])

        return {
            "score": scores,
            "identifier": self.model_identifier,
            "model_path_or_url": self.model_path_or_url,
            "lang": self.lang,
            "segment_scores": segment_scores,
            "normalized": normalize,
        }

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        segment_scores: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        self._load_scorer()
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                pred = [pred] * len(refs)
                prism_score = self._compute_prism_score(
                    predictions=pred, references=refs, segment_scores=True, **kwargs
                )
                reduced_pred_score = float(reduce_fn(prism_score))
                pred_scores.append(reduced_pred_score)
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)

        if not segment_scores:
            scores = sum(scores) / len(scores)

        if normalize:
            scores = self._normalize_score(scores, self.model_identifier["log_base"])

        return {
            "score": scores,
            "identifier": self.model_identifier,
            "model_path_or_url": self.model_path_or_url,
            "lang": self.lang,
            "segment_scores": segment_scores,
            "normalized": normalize,
        }
