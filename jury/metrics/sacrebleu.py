from typing import Dict

from jury.metrics import Metric

__class_name__ = "SacreBleu"


class SacreBleu(Metric):
	def __init__(self, resulting_name: str, params: Dict = None):
		resulting_name = "sacrebleu" if resulting_name is None else resulting_name
		super().__init__(metric_name="rouge", resulting_name=resulting_name, params=params)

	def _postprocess(self, result):
		score = result["rougeL"].mid.fmeasure
		return {self.resulting_name: score}
