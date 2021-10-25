# coding=utf-8
# Copyright 2020 Open Business Software Solutions, The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

import importlib
import os
import warnings
from typing import Any, Dict, NamedTuple, Optional

import datasets

from jury.metrics._core.base import Metric
from jury.metrics._core.utils import import_module, list_metrics


def load_metric(
    metric_name: str,
    resulting_name: Optional[str] = None,
    task: Optional[str] = None,
    compute_kwargs: Optional[Dict[str, Any]] = None,
    use_jury_only: bool = False,
    **kwargs,
) -> Metric:
    """Load a :py:class:`jury.metrics.Metric`. Alias for :py:class:`jury.metrics.AutoMetric.load()`.

    Args:

        metric_name (``str``):
            path to the metric processing script with the metric builder. Can be either:
                - a local absolute or relative path to processing script or the directory containing the script,
                    e.g. ``'./metrics/rogue/rouge.py'``
                - a metric identifier on the HuggingFace datasets repo (list all available metrics with
                    ``jury.list_metrics()``) e.g. ``'rouge'`` or ``'bleu'``
        resulting_name (Optional ``str``): Resulting name of the computed score returned.
        task (Optional ``str``): Task name for the metric. "language-generation" by default.
        compute_kwargs (Optional ``Dict[str, Any]``): Arguments to be passed to `compute()` method of metric at
            computation.
        use_jury_only (``bool``): Whether to use jury metrics only or not. False by default.
        kwargs (Optional): Additional keyword arguments to be passed to :py:func:`datasets.load_metric`.

    Returns:
        `datasets.Metric`
    """
    return AutoMetric.load(
        metric_name=metric_name,
        resulting_name=resulting_name,
        task=task,
        use_jury_only=use_jury_only,
        compute_kwargs=compute_kwargs,
        **kwargs,
    )


class AutoMetric:
    """
    Instantiates the proper metric class from given parameters.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        raise EnvironmentError("This class is designed to be instantiated by using 'from_params()' method.")

    @classmethod
    def load(
        cls,
        metric_name: str,
        task: Optional[str] = None,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        use_jury_only: bool = False,
        **kwargs,
    ) -> Metric:
        resolved_metric_name = cls.resolve_metric_name(metric_name)
        if task is None:
            task = "language-generation"

        # load the module, will raise ImportError if module cannot be loaded
        try:
            module_path = resolved_metric_name.path
            if resolved_metric_name.resolution == "external-module":
                module_name = module_path.split("/")[-1].replace(".py", "")
                module = import_module(module_name=module_name, filepath=module_path)
            else:
                module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            # Metric not in Jury
            if use_jury_only:
                raise ValueError(
                    f"Metric {resolved_metric_name.path} is not available on jury, set use_jury_only=False to use"
                    f"additional metrics (e.g datasets metrics)."
                )
            warnings.warn(
                f"Metric {resolved_metric_name.path} is not available on jury, falling back to datasets metric. "
                f"You may not fully utilize this metric for different input types, e.g multiple predictions"
                f"or multiple references."
            )
            metric = datasets.load_metric(resolved_metric_name.path, **kwargs)
        else:
            # get the class, will raise AttributeError if class cannot be found
            factory_class = module.__main_class__
            klass = getattr(module, factory_class)
            metric = klass.construct(task=task, resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)
        return metric

    @staticmethod
    def resolve_metric_name(metric_name: str):
        class ResolvedName(NamedTuple):
            path: str
            resolution: str

        if metric_name in list_metrics():
            metric_name = metric_name.lower()
            module_name = f"jury.metrics.{metric_name}.{metric_name}"
            return ResolvedName(path=module_name, resolution="internal-module")
        elif os.path.exists(metric_name):
            return ResolvedName(path=metric_name, resolution="external-module")
        else:
            return ResolvedName(path=metric_name, resolution="datasets")
