# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors
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
Metrics base class. The part of this file is adapted from HuggingFace's
datasets package implementation of Accuracy metric. See
https://github.com/huggingface/datasets/blob/master/src/datasets/metric.py
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy
import numpy as np
import pandas as pd
from datasets.utils.logging import get_logger

from jury.collator import Collator
from jury.metrics._core.utils import import_module, is_reduce_fn

LanguageGenerationInstance = Union[List[str], List[List[str]]]
SequenceClassificationInstance = Union[List[int], List[List[int]]]
SequenceLabelingInstance = List[List[str]]
EvaluationInstance = Union[LanguageGenerationInstance, SequenceClassificationInstance, SequenceLabelingInstance]
MetricOutput = Dict[str, Union[str, int, float]]

logger = get_logger(__name__)


class Metric(datasets.Metric, ABC):
    """
    Base metric class and common API for all metrics.

    Args:
        task (``str``): Task for the metric to be used. Tasks differ in inputs of predictions or references.
        resulting_name (Optional ``[str]``): Optional resulting name for :py:class:`jury.Jury` to use. By default, it
            uses `metric.name` if not given. This is meant to prevent clashes for output dict of
            :py:method:`jury.Jury.evaluate` such as when bleu-1, and bleu-2 are used together.
        compute_kwargs (Optional ``Dict[str, Any]``): These are the parameters to be passed to compute function of the
            metric. It is meant to ease the support of computation from a jury configuration file, etc.
        config_name (Optional ``str``): This is used to define a hash specific to a metrics computation script and
            prevents the metric's data to be overridden when the metric loading script is modified.
        keep_in_memory (``bool``): keep all predictions and references in memory. Not possible in distributed settings.
        cache_dir (Optional ``str``): Path to a directory in which temporary prediction/references data will be stored.
            The data directory should be located on a shared file-system in distributed setups.
        num_process (``int``): specify the total number of nodes in a distributed settings.
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        process_id (``int``): specify the id of the current process in a distributed setup (between 0 and num_process-1)
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        seed (Optional ``int``): If specified, this will temporarily set numpy's random seed when
        :func:`datasets.Metric.compute` is run.
        experiment_id (Optional ``str``): A specific experiment id. This is used if several distributed evaluations
            share the same file system. This is useful to compute metrics in distributed setups (in particular
            non-additive metrics like F1).
        max_concurrent_cache_files (``int``): Max number of concurrent metrics cache files (default 10000).
        timeout (``Union[int, float]``): Timeout in second for distributed setting synchronization.
    """

    def __init__(
        self,
        task: str,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs,
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs,
        )
        self._task = task
        self.resulting_name = resulting_name if resulting_name is not None else self.name
        self.compute_kwargs = compute_kwargs or {}
        self.download_and_prepare()

    @abstractmethod
    def _compute(
        self,
        *,
        predictions: EvaluationInstance = None,
        references: EvaluationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        """
        Base compute method which is used for internal computation. All child classes
        must implement _compute() method.
        """
        pass

    @abstractmethod
    def _compute_single_pred_single_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        Args:
            predictions: (``EvaluationInstance``) Predictions
            references: (``EvaluationInstance``) References
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_single_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        Args:
            predictions: (``EvaluationInstance``) Predictions
            references: (``EvaluationInstance``) References
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_multi_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        Args:
            predictions: (``EvaluationInstance``) Predictions
            references: (``EvaluationInstance``) References
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    def _download_and_prepare(self, dl_manager):
        """
        Downloads and prepares resources for the metric.

        This is the internal implementation to overwrite called when user calls
        `download_and_prepare`. It should download all required resources for the metric.

        Args:
            dl_manager (:class:`DownloadManager`): `DownloadManager` used to download and cache data.
        """
        self.external_module_path = None
        return None

    def _get_external_resource(self, module_name: Optional[str], attr: Optional[str] = None):
        if self.external_module_path is None:
            raise AttributeError("'external_module_path' is not defined.")
        if module_name is None:
            module_name = "external_module"
        external_module = import_module(module_name, self.external_module_path)
        if attr is None:
            return external_module
        return getattr(external_module, attr)

    @property
    def task(self):
        return self._task


class MetricForTask(Metric):
    """
    Base metric class for any task. All metrics must extend this class as metric is required to adopt a task
    inherently. Default task will be language-generation for AutoMetric.

    All metrics extending :py:class:`jury.metrics._core.base.MetricForTask` must implement the following:

        - _task (``[str]``): Task name for the base task metric.
        - _default_features() (``datasets.Features``): Task input as a :py:class:`datasets.Features`.

     Args:
        resulting_name (Optional ``[str]``): Optional resulting name for :py:class:`jury.Jury` to use. By default, it
            uses `metric.name` if not given. This is meant to prevent clashes for output dict of
            :py:method:`jury.Jury.evaluate` such as when bleu-1, and bleu-2 are used together.
        compute_kwargs (Optional ``Dict[str, Any]``): These are the parameters to be passed to compute function of the
            metric. It is meant to ease the support of computation from a jury configuration file, etc.
    """

    _task = None

    def __init__(self, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        compute_kwargs = self._validate_compute_kwargs(compute_kwargs)
        super().__init__(task=self._task, resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _validate_compute_kwargs(self, compute_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if compute_kwargs is not None and "reduce_fn" in compute_kwargs:
            compute_kwargs.pop("reduce_fn")
        return compute_kwargs

    @property
    def _default_features(self):
        raise NotImplementedError

    @classmethod
    def _construct(
        cls, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        return cls(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @staticmethod
    def _reduce_scores(scores: Union[List[Dict[str, float]], List[float]], reduce_fn: Callable):
        if isinstance(scores[0], dict):
            score = pd.DataFrame(scores).apply(reduce_fn, axis=0).to_dict()
        else:
            score = float(reduce_fn(scores))
        return score

    def _preprocess(self, predictions: List[List[str]], references: List[List[str]]) -> Tuple[Collator, Collator]:
        return Collator(predictions, keep=True), Collator(references, keep=True)

    def _compute(
        self,
        *,
        predictions: EvaluationInstance = None,
        references: EvaluationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        assert len(predictions) == len(references), "Predictions and references length does not match."
        eval_params = {**self.compute_kwargs, **kwargs}
        if "reduce_fn" in eval_params:
            eval_params.pop("reduce_fn")
        predictions, references = Collator(predictions), Collator(references)
        result = self.evaluate(predictions=predictions, references=references, **eval_params)
        return {self.resulting_name: result}

    def evaluate(self, predictions: Collator, references: Collator, **kwargs) -> Dict[str, float]:
        if predictions.can_collapse() and references.can_collapse():
            predictions = predictions.collapse()
            references = references.collapse()
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref

        predictions, references = self._preprocess(predictions, references)
        return eval_fn(predictions=predictions, references=references, **kwargs)


class MetricForLanguageGeneration(MetricForTask):
    """
    Base metric class for language generation task. Many metrics on jury are language generation metrics which are
    used by default by :py:class:`jury.metrics.AutoMetric`.
    """

    _task = "language-generation"

    @property
    def _default_features(self):
        return datasets.Features(
            {
                "predictions": datasets.Sequence(datasets.Value("string", id="sequence")),
                "references": datasets.Sequence(datasets.Value("string", id="sequence")),
            }
        )

    def _validate_compute_kwargs(self, compute_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if compute_kwargs is None:
            compute_kwargs = {}
        if "reduce_fn" not in compute_kwargs:
            compute_kwargs.update({"reduce_fn": "max"})
        return compute_kwargs

    def _compute(
        self,
        *,
        predictions: LanguageGenerationInstance = None,
        references: LanguageGenerationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        assert len(predictions) == len(references), "Predictions and references length does not match."
        reduce_fn = kwargs.get("reduce_fn")
        reduce_fn = self.compute_kwargs["reduce_fn"] if reduce_fn is None else reduce_fn
        if isinstance(reduce_fn, str):
            reduce_fn = getattr(numpy, reduce_fn)
        elif reduce_fn is not None and not callable(reduce_fn):
            raise TypeError(f"'reduce_fn' Expected str or callable, got {type(reduce_fn)}")
        if reduce_fn is not None and not is_reduce_fn(reduce_fn):
            raise ValueError("'reduce_fn' must be an aggregation function.")
        eval_params = {**self.compute_kwargs, **kwargs}
        eval_params.pop("reduce_fn")
        predictions, references = Collator(predictions), Collator(references)
        result = self.evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **eval_params)
        return {self.resulting_name: result}

    @abstractmethod
    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        Args:
            predictions: (``List[str]``) Predictions
            references: (``List[str]``) References
            reduce_fn: (``Callable``) Reduce function name.
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        Args:
            predictions: (``List[str]``) Predictions
            references: (``List[List[str]]``) References
            reduce_fn: (``Callable``) Reduce function name.
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        Args:
            predictions: (``List[List[str]]``) Predictions
            references: (``List[List[str]]``) References
            reduce_fn: (``Callable``) Reduce function name.
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass


class MetricForSequenceClassification(MetricForTask):
    """
    Base metric class for sequence classification task.
    """

    _task = "sequence-classification"

    @property
    def _default_features(self):
        return datasets.Features(
            {
                "predictions": datasets.Sequence(datasets.Value("int32")),
                "references": datasets.Sequence(datasets.Value("int32")),
            }
        )

    @abstractmethod
    def _compute_single_pred_single_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        Args:
            predictions: (``List[int]``) Predictions
            references: (``List[int]``) References
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_single_pred_multi_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def _compute_multi_pred_multi_ref(
        self,
        predictions: SequenceClassificationInstance,
        references: SequenceClassificationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        pass

    @staticmethod
    def _get_class_ids(references: List[List]):
        stacked_references = []
        for ref in references:
            stacked_references.extend(ref)
        return np.unique(stacked_references).tolist()


class MetricForSequenceLabeling(MetricForTask):
    """
    Base metric class for sequence labeling task.
    """

    _task = "sequence-labeling"

    @property
    def _default_features(self):
        return datasets.Features(
            {
                "predictions": datasets.Sequence(datasets.Value("string", id="sequence")),
                "references": datasets.Sequence(datasets.Value("string", id="sequence")),
            }
        )

    @abstractmethod
    def _compute_single_pred_single_ref(
        self,
        predictions: SequenceLabelingInstance,
        references: SequenceLabelingInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        Args:
            predictions: (``List[List[str]]``) Predictions
            references: (``List[List[str]]``) References
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    def _compute_single_pred_multi_ref(
        self,
        predictions: SequenceLabelingInstance,
        references: SequenceLabelingInstance,
        **kwargs,
    ):
        raise NotImplementedError(f"Task {self._task} does not support multiple predictions or multiple" "references.")

    def _compute_multi_pred_multi_ref(
        self,
        predictions: SequenceLabelingInstance,
        references: SequenceLabelingInstance,
        **kwargs,
    ):
        raise NotImplementedError(f"Task {self._task} does not support multiple predictions or multiple" "references.")
