from abc import ABC
from typing import Dict, Union

import datasets


class Metric(ABC):
    def __init__(self, metric_name: str, resulting_name: str = None, params: Dict = None):
        self.metric_name = metric_name.lower()
        self.resulting_name = resulting_name if resulting_name is not None else metric_name
        self.params = params if params is not None else {}

    def compute(self, predictions, references, return_dict: bool = True, fn_multiple: bool = False):
        predictions, references = self._preprocess(predictions, references, fn_multiple=fn_multiple)
        metric = datasets.load_metric(self.metric_name)
        result = metric.compute(predictions=predictions, references=references, **self.params)
        return self._postprocess(result, return_dict=return_dict)

    def _preprocess(self, predictions, references, fn_multiple):
        return predictions, references

    def _postprocess(self, result, return_dict: bool) -> Union[Dict[str, float], float]:
        score = result[self.metric_name]
        if not return_dict:
            return score
        return {self.resulting_name: float(score)}
