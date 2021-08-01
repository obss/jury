from typing import Dict

from jury.collator import Collator
from jury.metrics import Metric

__class_name__ = "SQUAD"


class SQUAD(Metric):
    def __init__(self, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__
        resulting_name = metric_name if resulting_name is None else resulting_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["f1"] / 100}
        return super()._postprocess(result, return_dict)

    def _preprocess(self, predictions, references):
        predictions = [{"prediction_text": pred, "id": str(i)} for i, pred in enumerate(predictions.collapse())]
        references = [
            {"answers": {"answer_start": [-1], "text": Collator(ref, keep=True).collapse()}, "id": str(i)}
            for i, ref in enumerate(references)
        ]
        return predictions, references
