import importlib
from typing import Dict

from jury.metrics.metric import Metric


def load_metric(metric_name: str, resulting_name: str = None, params: Dict = None):
    # load the module, will raise ImportError if module cannot be loaded
    module_name = f"jury.metrics.{metric_name.lower()}"
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, m.__class_name__)
    return c(resulting_name=resulting_name, params=params)
