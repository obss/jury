from typing import Dict

from jury.metrics import Metric

__class_name__ = "Meteor"


class Meteor(Metric):
	def __init__(self, resulting_name: str = None, params: Dict = None):
		super().__init__(metric_name="meteor", resulting_name=resulting_name, params=params)
