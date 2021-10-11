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

import importlib
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy
import pandas as pd
from datasets.utils.logging import get_logger

from jury.collator import Collator
from jury.metrics._utils import import_module, is_reduce_fn

logger = get_logger(__name__)


def load_metric(metric_name: str, resulting_name: str = None, params: Dict = None) -> "Metric":
    # load the module, will raise ImportError if module cannot be loaded
    metric_name = metric_name.lower()
    module_name = f"jury.metrics.{metric_name}"
    try:
        m = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Metric not in Jury
        warnings.warn(
            f"Metric {metric_name} is not available on Jury, falling back to datasets metric. "
            f"You may not fully utilize this metric for different input types, e.g multiple predictions"
            f"or multiple references."
        )
        metric = datasets.load_metric(metric_name)
    else:
        # get the class, will raise AttributeError if class cannot be found
        c = getattr(m, m.__class_names__.get(metric_name))
        metric = c(resulting_name=resulting_name, params=params)
    return metric


class Metric(datasets.Metric, ABC):
    """Base metric class and common API for all metrics.

    Args:
        resulting_name (``Optional[str]``): Optional resulting name for :py:class:`jury.Jury` to use. By default, it
            uses `metric.name` if not given. This is meant to prevent clashes for output dict of
            :py:method:`jury.Jury.evaluate` such as when bleu-1, and bleu-2 are used together.
        params (``Optional[Dict[str, Any]]``): These are the parameters to be passed to compute function of the metric.
            It is meant to ease the support of computation from a jury configuration file, etc.
        config_name (``str``): This is used to define a hash specific to a metrics computation script and prevents the metric's data
            to be overridden when the metric loading script is modified.
        keep_in_memory (``bool``): keep all predictions and references in memory. Not possible in distributed settings.
        cache_dir (``str``): Path to a directory in which temporary prediction/references data will be stored.
            The data directory should be located on a shared file-system in distributed setups.
        num_process (``int``): specify the total number of nodes in a distributed settings.
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        process_id (``int``): specify the id of the current process in a distributed setup (between 0 and num_process-1)
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        seed (Optional ``int``): If specified, this will temporarily set numpy's random seed when :func:`datasets.Metric.compute` is run.
        experiment_id (``str``): A specific experiment id. This is used if several distributed evaluations share the same file system.
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        max_concurrent_cache_files (``int``): Max number of concurrent metrics cache files (default 10000).
        timeout (``Union[int, float]``): Timeout in second for distributed setting synchronization.
    """

    default_features = datasets.Features(
        {
            "predictions": datasets.Sequence(datasets.Value("string", id="sequence")),
            "references": datasets.Sequence(datasets.Value("string", id="sequence")),
        }
    )

    def __init__(self, resulting_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.resulting_name = resulting_name if resulting_name is not None else self.name
        self.params = params or {}
        if "reduce_fn" not in self.params:
            self.params.update({"reduce_fn": "max"})
        self.download_and_prepare()

    def _compute(self, predictions: List[List[str]], references: List[List[str]], **kwargs) -> Dict[str, float]:
        assert len(predictions) == len(references), "Predictions and references length does not match."
        reduce_fn = kwargs.get("reduce_fn")
        reduce_fn = self.params["reduce_fn"] if reduce_fn is None else reduce_fn
        if isinstance(reduce_fn, str):
            reduce_fn = getattr(numpy, reduce_fn)
        elif reduce_fn is not None and not callable(reduce_fn):
            raise TypeError(f"'reduce_fn' Expected str or callable, got {type(reduce_fn)}")
        if reduce_fn is not None and not is_reduce_fn(reduce_fn):
            raise ValueError("'reduce_fn' must be an aggregation function.")
        eval_params = {**self.params, **kwargs}
        eval_params.pop("reduce_fn")
        predictions, references = Collator(predictions), Collator(references)
        result = self.evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **eval_params)
        return {self.resulting_name: result}

    @abstractmethod
    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        Args:
            predictions: (``List[str]``) Predictions
            references: (``List[str]``) References
            reduce_fn: (``str``) Reduce function name.
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        Args:
            predictions: (``List[str]``) Predictions
            references: (``List[List[str]]``) References
            reduce_fn: (``str``) Reduce function name.
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    @abstractmethod
    def _compute_multi_pred_multi_ref(self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        Args:
            predictions: (``List[List[str]]``) Predictions
            references: (``List[List[str]]``) References
            reduce_fn: (``str``) Reduce function name.
            **kwargs: Additional arguments used for the metric computation.

        Returns: score
        """
        pass

    def _download_and_prepare(self, dl_manager):
        """Downloads and prepares resources for the metric.

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

    @staticmethod
    def _reduce_scores(scores: Union[List[Dict[str, float]], List[float]], reduce_fn: Callable):
        if isinstance(scores[0], dict):
            score = pd.DataFrame(scores).apply(reduce_fn, axis=0).to_dict()
        else:
            score = float(reduce_fn(scores))
        return score

    def _preprocess(self, predictions: List[List[str]], references: List[List[str]]) -> Tuple[Collator, Collator]:
        return Collator(predictions, keep=True), Collator(references, keep=True)

    def evaluate(
        self, predictions: Collator, references: Collator, reduce_fn: Optional[Callable] = None, **kwargs
    ) -> Dict[str, float]:
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
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
