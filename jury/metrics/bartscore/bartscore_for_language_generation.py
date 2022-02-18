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
""" Bartscore metric. The part of this file is adapted from metric implementations
of datasets package. See
https://github.com/huggingface/datasets/blob/master/metrics/ """
import os
import warnings
from typing import Callable, Dict, List, Union

import datasets
import numpy as np

from jury.metrics import LanguageGenerationInstance
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import download

_CITATION = """
@misc{yuan2021bartscore,
      title={BARTScore: Evaluating Generated Text as Text Generation}, 
      author={Weizhe Yuan and Graham Neubig and Pengfei Liu},
      year={2021},
      eprint={2106.11520},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
[TODO: DESCRIPTION, COULDNT FIND AN EASY ONE]

See the `README.md` file at [https://github.com/neulab/BARTScore](https://github.com/neulab/BARTScore) for more
information.
"""

_KWARGS_DESCRIPTION = """
Prism metric arguments.

Construction Args:
    model_checkpoint (str): [TODO]
    model_path_or_url (str): Path to the model directory or a URL of model file (pth).
    device (str): On which the contextual embedding model will be allocated on.
        If this argument is None, the model lives on cuda:0 if cuda is available.
    
Computation Args:
    predictions (list of str): Prediction/candidate sentences.
    references (list of str or list of list of str): Reference sentences.
    batch_size (int): BARTScore processing batch size.
    segment_scores (bool): If True, then score for each instance are returned separately. Otherwise,
        average score is returned.

Returns:
    'score': BARTScore loss.

Examples:

    >>> bartscore = jury.load_metric("bartscore")
    >>> predictions = [
        ["the cat is on the mat", "There is cat playing on mat"],
        ["Look! what a wonderful day, today.", "Today is a very wonderful day"],
    ]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]
    >>> results = bartscore.compute(predictions=predictions, references=references)
    >>> print(results)
    {
      "score": -1.1489432752132416,
      "identifier": {
          "version": "0.1",
          "model": "m39v1",
          "seg_scores": "avg_log_prob",
          "sys_scores": "avg_log_prob",
          "log_base": 2,
          "temperature": 1.0
      },
      "model_path_or_url": "http://data.statmt.org/prism/m39v1.tar",
      "lang": "en",
      "segment_scores": false,
      "normalized": false
    }
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BartscoreForLanguageGeneration(MetricForLanguageGeneration):
    def __init__(
        self,
        resulting_name: str = None,
        compute_kwargs: Dict = None,
        model_checkpoint: str = "facebook/bart-large-cnn",
        model_path_or_url: str = None,
        max_length: int = 1024,
        device: str = None,
        **kwargs,
    ):
        self.model_checkpoint = model_checkpoint
        self.model_path_or_url = model_path_or_url
        self.max_length = max_length
        self.device = device
        self.model_path = None
        self.scorer = None
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of BARTScore score from the implementation
        of BARTScore computation from neulab/BARTScore. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/neulab/BARTScore/blob/47b8341854e1b8be965b65480ce236b0c2f7543b/bart_score.py
        """
        self.model_path = dl_manager.download(self.model_path_or_url) if self.model_path_or_url else None
        bartscore_source = (
            "https://raw.githubusercontent.com/neulab/BARTScore/47b8341854e1b8be965b65480ce236b0c2f7543b/bart_score.py"
        )
        bartscore_dest = dl_manager.download(bartscore_source)
        self.external_module_path = bartscore_dest
        BARTScorer = self._get_external_resource("bart_score", attr="BARTScorer")
        self.scorer = BARTScorer(device=self.device, max_length=self.max_length, checkpoint=self.model_checkpoint)
        if self.model_path is not None:
            self.scorer.load(path=self.model_path)

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/neulab/BARTScore",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/neulab/BARTScore"],
            reference_urls=[
                "https://github.com/neulab/BARTScore",
                "https://arxiv.org/abs/2106.11520",
            ],
        )

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
        batch_size: int = 4,
        segment_scores: bool = False,
        **kwargs,
    ):
        score = self.scorer.score(predictions, references, batch_size=batch_size, **kwargs)
        if not segment_scores:
            score = np.mean(score)

        return {
            "score": score,
            "model_checkpoint": self.model_checkpoint,
            "model_path_or_url": self.model_path_or_url,
            "segment_scores": segment_scores,
        }

    # This is the multi_ref_score from BARTScore, but modified to accept any reduce function
    def _generic_multi_ref_score(self, srcs, tgts: List[List[str]], reduce_fn: Callable, batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.scorer.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        score_list = reduce_fn(score_matrix, axis=0)
        return list(score_list)

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn=None,
        batch_size: int = 4,
        segment_scores: bool = False,
        **kwargs,
    ):
        if reduce_fn in [np.max, np.mean]:
            agg = "max" if reduce_fn == np.max else "mean"
            score = self.scorer.multi_ref_score(predictions, references, batch_size=batch_size, agg=agg, **kwargs)
        else:
            warnings.warn(
                f"Using '{reduce_fn.__name__}' as a non-native reduction function with BARTScore. "
                f"Only 'max' and 'mean' functions are natively supported."
            )
            score = self._generic_multi_ref_score(
                predictions, references, batch_size=batch_size, reduce_fn=reduce_fn, **kwargs
            )

        if not segment_scores:
            score = np.mean(score)

        return {
            "score": score,
            "model_checkpoint": self.model_checkpoint,
            "model_path_or_url": self.model_path_or_url,
            "segment_scores": segment_scores,
        }

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn=None,
        batch_size: int = 4,
        segment_scores: bool = False,
        **kwargs,
    ):
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

        return {
            "score": scores,
            "model_checkpoint": self.model_checkpoint,
            "model_path_or_url": self.model_path_or_url,
            "segment_scores": segment_scores,
        }
