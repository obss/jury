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
SeqEval metric. The part of this file is adapted from seqeval implementation
of datasets package. See
https://github.com/huggingface/datasets/blob/master/metrics/seqeval/seqeval.py
"""
import importlib
from typing import List, Optional, Union

import datasets

from jury.metrics._core import MetricForSequenceLabeling, SequenceClassificationInstance, SequenceLabelingInstance
from jury.metrics._core.utils import PackagePlaceholder, requirement_message

# `import seqeval` placeholder
seqeval = PackagePlaceholder(version="1.2.2")


_CITATION = """\
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
@misc{seqeval,
  title={{seqeval}: A Python framework for sequence labeling evaluation},
  url={https://github.com/chakki-works/seqeval},
  note={Software available from https://github.com/chakki-works/seqeval},
  author={Hiroki Nakayama},
  year={2018},
}
"""

_DESCRIPTION = """\
seqeval is a Python framework for sequence labeling evaluation.
seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

This is well-tested by using the Perl script conlleval, which can be used for
measuring the performance of a system that has processed the CoNLL-2000 shared task data.

seqeval supports following formats:
IOB1
IOB2
IOE1
IOE2
IOBES

See the [README.md] file at https://github.com/chakki-works/seqeval for more information.
"""

_KWARGS_DESCRIPTION = """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.

Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
    suffix: True if the IOB prefix is after type, False otherwise. default: False
    scheme: Specify target tagging scheme. Should be one of ["IOB1", "IOB2", "IOE1", "IOE2", "IOBES", "BILOU"].
        default: None
    mode: Whether to count correct entity labels with incorrect I/B tags as true positives or not.
        If you want to only count exact matches, pass mode="strict". default: None.
    sample_weight: Array-like of shape (n_samples,), weights for individual samples. default: None
    zero_division: Which value to substitute as a metric value when encountering zero division. Should be on of 0, 1,
        "warn". "warn" acts as 0, but the warning is raised.

Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall:
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:

    >>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> seqeval = datasets.load_metric("seqeval")
    >>> results = seqeval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['MISC', 'PER', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']
    >>> print(results["overall_f1"])
    0.5
    >>> print(results["PER"]["f1"])
    1.0
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SeqevalForLanguageGeneration(MetricForSequenceLabeling):
    def _download_and_prepare(self, dl_manager):
        global seqeval
        try:
            import seqeval.metrics
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(metric_name="Seqeval", package_name="seqeval"))
        else:
            super(SeqevalForLanguageGeneration, self)._download_and_prepare(dl_manager)

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/chakki-works/seqeval",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/chakki-works/seqeval"],
            reference_urls=["https://github.com/chakki-works/seqeval"],
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: SequenceLabelingInstance,
        references: SequenceLabelingInstance,
        suffix: bool = False,
        scheme: Optional[str] = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = "warn",
    ):
        if scheme is not None:
            try:
                scheme_module = importlib.import_module("seqeval.scheme")
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")
        report = seqeval.metrics.classification_report(
            y_true=references,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        scores = {
            type_name: {
                "precision": float(score["precision"]),
                "recall": float(score["recall"]),
                "f1": float(score["f1-score"]),
                "number": int(score["support"]),
            }
            for type_name, score in report.items()
        }
        scores["overall_precision"] = float(overall_score["precision"])
        scores["overall_recall"] = float(overall_score["recall"])
        scores["overall_f1"] = float(overall_score["f1-score"])
        scores["overall_accuracy"] = float(seqeval.metrics.accuracy_score(y_true=references, y_pred=predictions))

        return scores

    def _compute_single_pred_multi_ref(
        self,
        predictions: SequenceLabelingInstance,
        references: SequenceLabelingInstance,
        **kwargs,
    ):
        return self._compute_single_pred_single_ref(predictions=predictions, references=references, **kwargs)

    def _compute_multi_pred_multi_ref(
        self,
        predictions: SequenceLabelingInstance,
        references: SequenceLabelingInstance,
        **kwargs,
    ):
        return self._compute_single_pred_single_ref(predictions=predictions, references=references, **kwargs)
