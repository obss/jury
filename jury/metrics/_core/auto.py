import importlib
import os
import warnings
from typing import Any, Dict, NamedTuple, Optional

import datasets

from jury.metrics._core.base import Metric
from jury.metrics._core.utils import import_module, list_metric_modules


def load_metric(
    metric_name: str,
    resulting_name: str = None,
    task: Optional[str] = "language-generation",
    compute_kwargs: Dict[str, Any] = None,
    use_jury_only: bool = False,
    **kwargs,
) -> Metric:
    return AutoMetric.from_params(
        metric_name=metric_name,
        resulting_name=resulting_name,
        task=task,
        use_jury_only=use_jury_only,
        compute_kwargs=compute_kwargs,
        **kwargs,
    )


class AutoMetric:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        raise EnvironmentError("This class is designed to be instantiated by using 'from_params()' method.")

    @classmethod
    def from_params(
        cls,
        metric_name: str,
        task: Optional[str] = "language-generation",
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        use_jury_only: bool = False,
        **kwargs,
    ) -> Metric:
        resolved_metric_name = cls.resolve_metric_name(metric_name)

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
            metric = datasets.load_metric(resolved_metric_name.path)
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

        if metric_name in list_metric_modules():
            metric_name = metric_name.lower()
            module_name = f"jury.metrics.{metric_name}.{metric_name}"
            return ResolvedName(path=module_name, resolution="internal-module")
        elif os.path.exists(metric_name):
            return ResolvedName(path=metric_name, resolution="external-module")
        else:
            return ResolvedName(path=metric_name, resolution="datasets")