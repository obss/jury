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
Comet cross-lingual MT evaluation metric. The part of this file is adapted from
Comet implementation of evaluate package. See
https://github.com/huggingface/evaluate/blob/master/metrics/comet/comet.py
"""

from typing import Callable, Union

import evaluate

from jury.metrics import LanguageGenerationInstance
from jury.metrics._core import MetricForCrossLingualEvaluation
from jury.metrics._core.utils import PackagePlaceholder, requirement_message

# `import comet` placeholder
comet = PackagePlaceholder(version="1.0.1")

_CITATION = """\
@inproceedings{rei-EtAl:2020:WMT,
   author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
   title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
   booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
   month          = {November},
   year           = {2020},
   address        = {Online},
   publisher      = {Association for Computational Linguistics},
   pages     = {909--918},
}
@inproceedings{rei-etal-2020-comet,
   title = "{COMET}: A Neural Framework for {MT} Evaluation",
   author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
   booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
   month = nov,
   year = "2020",
   address = "Online",
   publisher = "Association for Computational Linguistics",
   url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
   pages = "2685--2702",
}
"""

_DESCRIPTION = """\
Crosslingual Optimized Metric for Evaluation of Translation (COMET) is an open-source framework used to train 
Machine Translation metrics that achieve high levels of correlation with different types of human judgments 
(HTER, DA's or MQM). With the release of the framework the authors also released fully trained models that were used 
to compete in the WMT20 Metrics Shared Task achieving SOTA in that years competition. See the [README.md] file at 
https://unbabel.github.io/COMET/html/models.html for more information.
"""

_KWARGS_DESCRIPTION = """
COMET score.
Args:
`sources` (list of str): Source sentences
`predictions` (list of str): candidate translations
`references` (list of str): reference translations
`cuda` (bool): If set to True, runs COMET using GPU
`show_progress` (bool): Shows progress
`model`: COMET model to be used. Will default to `wmt-large-da-estimator-1719` if None.
Returns:
    `samples`: List of dictionaries with `src`, `mt`, `ref` and `score`.
    `scores`: List of scores.
Examples:
    >>> comet_metric = jury.load_metric('comet', config_name="wmt21-cometinho-da")
    >>> source = ["Die Katze spielt auf der Matte.", "Heute ist ein wunderbarer Tag."]
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = comet_metric.compute(sources=source, predictions=hypothesis, references=reference)
    >>> print(results)
    {'comet': {'scores': [0.6338749527931213, 0.4925243854522705], 'samples': 0.5631996691226959}}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CometForCrossLingualEvaluation(MetricForCrossLingualEvaluation):
    def _download_and_prepare(self, dl_manager):
        global comet
        try:
            import comet
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="comet", package_name="unbabel-comet"))
        else:
            super(CometForCrossLingualEvaluation, self)._download_and_prepare(dl_manager)

        if self.config_name == "default":
            checkpoint_path = comet.download_model("wmt20-comet-da")
        else:
            checkpoint_path = comet.download_model(self.config_name)
        self.scorer = comet.load_from_checkpoint(checkpoint_path)

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://unbabel.github.io/COMET/html/index.html",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/Unbabel/COMET"],
            reference_urls=[
                "https://github.com/Unbabel/COMET",
                "https://www.aclweb.org/anthology/2020.emnlp-main.213/",
                "http://www.statmt.org/wmt20/pdf/2020.wmt-1.101.pdf6",
            ],
        )

    def _compute_single_pred_single_ref(
        self,
        sources: LanguageGenerationInstance,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: Union[int, bool] = False,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ):
        data = {"src": sources, "mt": predictions, "ref": references}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        scores, samples = self.scorer.predict(
            data,
            batch_size=batch_size,
            gpus=gpus,
            mc_dropout=mc_dropout,
            progress_bar=progress_bar,
            accelerator=accelerator,
            num_workers=num_workers,
            length_batching=length_batching,
        )
        return {"scores": scores, "samples": samples}

    def _compute_single_pred_multi_ref(
        self,
        sources: LanguageGenerationInstance,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: Union[int, bool] = False,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ):
        scores = []
        for src, pred, refs in zip(sources, predictions, references):
            data = {"src": [src] * len(refs), "mt": [pred] * len(refs), "ref": refs}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            pred_scores, _ = self.scorer.predict(
                data,
                batch_size=batch_size,
                gpus=gpus,
                mc_dropout=mc_dropout,
                progress_bar=progress_bar,
                accelerator=accelerator,
                num_workers=num_workers,
                length_batching=length_batching,
            )
            scores.append(float(reduce_fn(pred_scores)))

        return {"scores": scores, "samples": sum(scores) / len(scores)}

    def _compute_multi_pred_multi_ref(
        self,
        sources: LanguageGenerationInstance,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: Union[int, bool] = False,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ):
        scores = []
        for src, preds, refs in zip(sources, predictions, references):
            all_pred_scores = []
            for pred in preds:
                data = {"src": [src] * len(refs), "mt": [pred] * len(refs), "ref": refs}
                data = [dict(zip(data, t)) for t in zip(*data.values())]
                pred_scores, _ = self.scorer.predict(
                    data,
                    batch_size=batch_size,
                    gpus=gpus,
                    mc_dropout=mc_dropout,
                    progress_bar=progress_bar,
                    accelerator=accelerator,
                    num_workers=num_workers,
                    length_batching=length_batching,
                )
                all_pred_scores.append(float(reduce_fn(pred_scores)))
            scores.append(float(reduce_fn(all_pred_scores)))

        return {"scores": scores, "samples": sum(scores) / len(scores)}
