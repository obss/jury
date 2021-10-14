import importlib
import os
import warnings
from typing import Any, Dict, NamedTuple, Optional

import datasets
from datasets import MetricInfo

from jury.metrics._core.base import EvaluationInstance, Metric, MetricOutput
from jury.metrics._core.utils import import_module


class AutoMetric(Metric):
    def _info(self) -> MetricInfo:
        raise NotImplementedError

    def __init__(
        self,
        metric_name: str,
        resulting_name: Optional[str] = None,
        task: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        use_jury_only: bool = True,
        **kwargs,
    ):
        super().__init__(task=task, resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)
        self.metric_name = metric_name
        self.use_jury_only = use_jury_only
        self.construction_kwargs = kwargs

    def _compute(
        self,
        *,
        predictions: EvaluationInstance = None,
        references: EvaluationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        metric = self.construct_metric()
        return metric.compute()

    def construct_metric(self):
        resolved_metric_name = self.resolve_metric_name()

        # load the module, will raise ImportError if module cannot be loaded
        try:
            module_path = resolved_metric_name.path
            if resolved_metric_name.resolution == "external-module" and not self.use_jury_only:
                module_name = module_path.split("/")[-1].replace(".py", "")
                module = import_module(module_name=module_name, filepath=module_path)
            else:
                module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            # Metric not in Jury
            if self.use_jury_only:
                raise ValueError(
                    f"Metric {resolved_metric_name.path} is not available on jury, set use_jury_only=False to use"
                    f"additional metrics (e.g datasets metrics)."
                )
            warnings.warn(
                f"Metric {resolved_metric_name.path} is not available on jury, falling back to datasets metric. "
                f"You may not fully utilize this metric for different input types, e.g multiple predictions"
                f"or multiple references."
            )
            metric = datasets.load_metric(resolved_metric_name.path)
        else:
            # get the class, will raise AttributeError if class cannot be found
            klass = getattr(module, module.__class_names__.get(resolved_metric_name.path))
            metric = klass(
                resulting_name=self.resulting_name, compute_kwargs=self.compute_kwargs, **self.construction_kwargs
            )
        return metric

    def resolve_metric_name(self):
        class ResolvedName(NamedTuple):
            path: str
            resolution: str

        metric_name = self.metric_name
        if os.path.exists(metric_name):
            return ResolvedName(path=metric_name, resolution="external-module")
        else:
            metric_name = metric_name.lower()
            module_name = f"jury.metrics.{metric_name}"
            return ResolvedName(path=module_name, resolution="internal-module")

    def _compute_single_pred_single_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        raise NotImplementedError

    def _compute_multi_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        raise NotImplementedError

    def _compute_single_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        raise NotImplementedError
