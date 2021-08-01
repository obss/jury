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


if __name__ == "__main__":
    m = load_metric("meteor")
    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
    res = m.compute(predictions=predictions, references=references)
    print(res)
