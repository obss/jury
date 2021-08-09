from typing import Dict

from jury.metrics import Metric

__class_names__ = {"bertscore": "BERTScore"}


class BERTScore(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        resulting_name = metric_name if resulting_name is None else resulting_name
        params = {"lang": "en"} if params is None else params
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _preprocess(self, predictions, references):
        predictions = predictions.to_list()
        references = references.to_list()
        return predictions, references

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["f1"][0]}
        return super()._postprocess(result, return_dict)
