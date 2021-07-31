from typing import Dict

from jury.metrics import Metric

__class_name__ = "Rouge"


class Rouge(Metric):
	def __init__(self, resulting_name: str = None, params: Dict = None):
		resulting_name = "rougeL" if resulting_name is None else resulting_name
		params = {"rouge_types": ["rougeL"]} if params is None else params
		super().__init__(metric_name="rouge", resulting_name=resulting_name, params=params)

	def _postprocess(self, result, return_dict: bool):
		result = {self.metric_name: result["rougeL"].mid.fmeasure}
		super()._postprocess(result, return_dict)
