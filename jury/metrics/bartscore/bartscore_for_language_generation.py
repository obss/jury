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
BARTScore metric. The part of this file is adapted from metric implementations
of evaluate package. See
https://github.com/huggingface/evaluate/blob/master/metrics/
"""
from typing import Callable, Dict, List

import evaluate
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
BARTScore formulates evaluating generated text as a text generation task from pre-trained language models. It
operationalizes this idea using BART, an encoder-decoder based pre-trained model. BARTScore is conceptually
simple and empirically effective.

See the `README.md` file at [https://github.com/neulab/BARTScore](https://github.com/neulab/BARTScore) for more
information.
"""

_KWARGS_DESCRIPTION = """
BARTScore metric arguments.

Construction Args:
    model_checkpoint (str): BARTScore checkpoint. Will default to bartscore-large-cnn.
    model_weights (str): Optional BARTScore weights, overrides the checkpoint weights.
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
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = bartscore.compute(predictions=predictions, references=references)
    >>> print(results)
    {
      "bartscore": {
        "score": -2.201135754585266,
        "model_checkpoint": "bartscore-large-cnn",
        "model_weights": null,
        "segment_scores": false
      }
    }
"""

_LICENSE = """Copyright 2022 NeuLab.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

CHECKPOINT_URLS = {
    "bartscore-large-cnn": {
        "model_checkpoint": "facebook/bart-large-cnn",
        "model_weights": {
            "parabank2": "https://drive.google.com/uc?export=download&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&confirm=t"
        },
    }
}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BartscoreForLanguageGeneration(MetricForLanguageGeneration):
    def __init__(
        self,
        resulting_name: str = None,
        compute_kwargs: Dict = None,
        model_checkpoint: str = "bartscore-large-cnn",
        model_weights: str = None,
        max_length: int = 1024,
        device: str = None,
        **kwargs,
    ):
        self.model_checkpoint = model_checkpoint
        self.model_weights = model_weights
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

        if self.model_checkpoint.lower() in CHECKPOINT_URLS:
            checkpoint_name = self.model_checkpoint.lower()
        else:
            raise KeyError(
                f"{self.model_checkpoint} checkpoint not found. You should supply the name of a model checkpoint for BARTScore in {CHECKPOINT_URLS.keys()}"
            )
        model_checkpoint = CHECKPOINT_URLS[checkpoint_name]["model_checkpoint"]

        if self.model_weights:
            if self.model_weights.lower() in CHECKPOINT_URLS[checkpoint_name]["model_weights"]:
                weights_name = self.model_weights.lower()
            else:
                raise KeyError(
                    f"Weights named '{self.model_weights}' not found. You should supply the name of a model weight for BARTScore in "
                    + str(list(CHECKPOINT_URLS[checkpoint_name]["model_weights"].keys()))
                )
            model_path = CHECKPOINT_URLS[checkpoint_name]["model_weights"][weights_name]
        else:
            model_path = None

        bartscore_source = (
            "https://raw.githubusercontent.com/neulab/BARTScore/47b8341854e1b8be965b65480ce236b0c2f7543b/bart_score.py"
        )
        self.external_module_path = dl_manager.download(bartscore_source)
        BARTScorer = self._get_external_resource("bart_score", attr="BARTScorer")

        self.scorer = BARTScorer(device=self.device, max_length=self.max_length, checkpoint=model_checkpoint)

        if model_path is not None:
            model_dest = dl_manager.download(model_path)
            self.scorer.load(path=model_dest)

    def _info(self):
        return evaluate.MetricInfo(
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
            license=_LICENSE,
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
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
            "model_weights": self.model_weights,
            "segment_scores": segment_scores,
        }

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
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
            "model_weights": self.model_weights,
            "segment_scores": segment_scores,
        }

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
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
            "model_weights": self.model_weights,
            "segment_scores": segment_scores,
        }
