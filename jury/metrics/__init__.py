import importlib
from typing import Dict

import datasets


def load_metric(metric_name: str, resulting_name: str = None, params: Dict = None):
	# load the module, will raise ImportError if module cannot be loaded
	module_name = f"jury.metrics.{metric_name}"
	m = importlib.import_module(module_name)
	# get the class, will raise AttributeError if class cannot be found
	c = getattr(m, m.__class_name__)
	return c(resulting_name=resulting_name, params=params)


class Metric:
	def __init__(self, metric_name: str, resulting_name: str = None, params: Dict = None):
		self.metric_name = metric_name
		self.resulting_name = resulting_name if resulting_name is not None else metric_name
		self.params = params if params is not None else {}
		self._metric = datasets.load_metric(metric_name)

	def compute(self, predictions, references, return_dict: bool = True):
		predictions, references = self._preprocess(predictions, references)
		result = self._metric.compute(predictions=predictions, references=references, **self.params)
		return self._postprocess(result, return_dict=return_dict)

	def _preprocess(self, predictions, references):
		return predictions, references

	def _postprocess(self, result, return_dict: bool):
		score = result[self.metric_name]
		if not return_dict:
			return score
		return {self.resulting_name: score}


if __name__ == "__main__":
	m = load_metric("meteor")
	predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
	references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
	res = m.compute(predictions=predictions, references=references)
	print(res)
