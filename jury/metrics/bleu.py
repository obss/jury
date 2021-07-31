from typing import Dict

from jury.metrics import Metric

__class_name__ = "Bleu"


class Bleu(Metric):
	def __init__(self, resulting_name: str = None, params: Dict = None):
		super().__init__(metric_name="bleu", resulting_name=resulting_name, params=params)
