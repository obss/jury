from typing import Dict

from jury.collator import Collator
from jury.metrics import Metric

__class_names__ = {"squad": "SQUAD_F1", "squad_f1": "SQUAD_F1", "squad_em": "SQUAD_EM"}

from jury.utils import NestedSingleType


class SQUAD(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        resulting_name = metric_name if resulting_name is None else resulting_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _preprocess(self, predictions, references):
        if NestedSingleType.get_type(predictions, order=-1) == "str":
            predictions = [{"prediction_text": pred, "id": str(i)} for i, pred in enumerate(predictions.collapse())]
        if NestedSingleType.get_type(references, order=-1) == "str":
            references = [
                {"answers": {"answer_start": [-1], "text": Collator(ref).collapse()}, "id": str(i)}
                for i, ref in enumerate(references)
            ]
        return predictions, references


class SQUAD_EM(SQUAD):
    def __init__(self, resulting_name: str = None, params: Dict = None):
        metric_name = "squad"
        resulting_name = self.__class__.__name__.lower() if resulting_name is None else resulting_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["exact_match"] / 100}
        return super()._postprocess(result, return_dict)


class SQUAD_F1(SQUAD):
    def __init__(self, resulting_name: str = None, params: Dict = None):
        metric_name = "squad"
        resulting_name = self.__class__.__name__.lower() if resulting_name is None else resulting_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["f1"] / 100}
        return super()._postprocess(result, return_dict)
