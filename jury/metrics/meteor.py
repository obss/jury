from typing import Dict

from jury.metrics import Metric

__class_names__ = {"meteor": "Meteor"}


class Meteor(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)
