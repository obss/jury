import importlib
from typing import Dict

from jury.metrics._base import Metric


def load_metric(metric_name: str, resulting_name: str = None, params: Dict = None):
    # load the module, will raise ImportError if module cannot be loaded
    metric_name = metric_name.lower()
    base_name = metric_name.split("_")[0]
    module_name = f"jury.metrics.{base_name}"
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, m.__class_names__.get(metric_name))
    return c(resulting_name=resulting_name, params=params)
