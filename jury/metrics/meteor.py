from typing import Dict

from jury.metrics import Metric

__class_name__ = "Meteor"


class Meteor(Metric):
    def __init__(self, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)
