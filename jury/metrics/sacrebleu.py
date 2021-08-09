from typing import Dict

from jury.metrics import Metric

__class_names__ = {"sacrebleu": "SacreBLEU"}


class SacreBLEU(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        resulting_name = metric_name if resulting_name is None else resulting_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["score"] / 100}
        return super()._postprocess(result, return_dict)

    def _preprocess(self, predictions, references):
        predictions = predictions.reshape(-1)
        references = references.reshape(1, -1)
        return predictions, references
