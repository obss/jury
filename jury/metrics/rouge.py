from typing import Dict

from jury.metrics import Metric

__class_names__ = {"rouge": "Rouge"}


class Rouge(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        resulting_name = "rougeL" if resulting_name is None else resulting_name
        params = {"rouge_types": ["rougeL"]} if params is None else params
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _postprocess(self, result, return_dict: bool):
        result = {self.metric_name: result["rougeL"].mid.fmeasure}
        return super()._postprocess(result, return_dict)
