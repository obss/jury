from typing import Dict

from jury.metrics import Metric

__class_name__ = "BERTScore"


class BERTScore(Metric):
	def __init__(self, resulting_name: str, params: Dict = None):
		metric_name = self.__class__.__name__
		resulting_name = metric_name if resulting_name is None else resulting_name
		params = {"lang": "en"} if params is None else params
		super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

	def _postprocess(self, result, return_dict):
		result = {self.metric_name: result["f1"][0]}
		return super()._postprocess(result, return_dict)
