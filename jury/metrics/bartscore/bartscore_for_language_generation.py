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
            score = float(np.mean(score))

        return {
            "score": score,
            "model_checkpoint": self.model_checkpoint,
            "model_path_or_url": self.model_path_or_url,
            "segment_scores": segment_scores,
        }

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn=None,
        batch_size: int = 4,
        segment_scores: bool = False,
        **kwargs,
    ):
        """
        Ideally we would use the already-implemented multi-ref function in BARTScore,
        but unfortunately it only supports two reduction functions and doesn't support
        inconsistent number of reference per predictions. So instead we pre-combine our
        prediction/reference pairs and use their single prediction/reference function
        in order to utilize the batch_size argument.
        """
        flat_predictions: List[str] = []
        flat_references: List[str] = []
        ranges = []
        cursor = 0
        for pred, refs in zip(predictions, references):
            ref_count = len(refs)
            flat_predictions += [pred] * ref_count
            flat_references += refs
            new_cursor = cursor + ref_count
            ranges.append((cursor, new_cursor))
            cursor = new_cursor

        flat_scores = self.scorer.score(flat_predictions, flat_references, batch_size=batch_size, **kwargs)
        score = []
        for start, end in ranges:
            reduced_score = float(reduce_fn(flat_scores[start:end]))
            score.append(reduced_score)

        if not segment_scores:
            score = float(np.mean(score))

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
        """
        Like Single Pred/Multi Ref, we pre-combine all possible prediction/reference
        pairs into a list of single prediction/single reference pairs in order to utilize
        the batch_size argument.
        """
        flat_predictions: List[str] = []
        flat_references: List[str] = []
        ranges = []
        shapes = []
        cursor = 0
        for preds, refs in zip(predictions, references):
            pred_count = len(preds)
            ref_count = len(refs)
            for pred in preds:
                flat_predictions += [pred] * ref_count
                flat_references += refs
            new_cursor = cursor + (pred_count * ref_count)
            ranges.append((cursor, new_cursor))
            shapes.append((pred_count, ref_count))
            cursor = new_cursor

        flat_scores = self.scorer.score(flat_predictions, flat_references, batch_size=batch_size, **kwargs)
        score = []
        for pair_range, pair_shape in zip(ranges, shapes):
            pair_start, pair_end = pair_range
            pred_count, ref_count = pair_shape
            pair_scores = []
            for i in range(pred_count):
                pred_start = pair_start + i * ref_count
                pred_end = pred_start + ref_count
                pred_score = flat_scores[pred_start:pred_end]
                reduced_pred_score = float(reduce_fn(pred_score))
                pair_scores.append(reduced_pred_score)
            reduced_score = float(reduce_fn(pair_scores))
            score.append(reduced_score)

        if not segment_scores:
            score = float(np.mean(score))

        return {
            "score": score,
            "model_checkpoint": self.model_checkpoint,
            "model_path_or_url": self.model_path_or_url,
            "segment_scores": segment_scores,
        }
